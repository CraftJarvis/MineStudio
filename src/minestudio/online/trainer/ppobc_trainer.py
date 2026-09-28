import shutil
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from minestudio.online.utils.train.data import prepare_batch, data_iter
from typing import Tuple, List, Optional
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment, FragmentDataDict
from minestudio.online.utils import auto_slice, recursive_detach
import minestudio.online.utils.train.wandb_logger as wandb_logger
from minestudio.models import MinePolicy
from minestudio.data import MineDataModule
from minestudio.offline.mine_callbacks import ObjectiveCallback
import time
import ray
import ray.train.torch
import os
import torchmetrics
import logging
from minestudio.online.trainer.ppotrainer import PPOTrainer
from ray.experimental import tqdm_ray
from minestudio.online.utils import auto_stack
import uuid
import copy
import torch.distributed as dist

def print_memory_usage():
    """Prints the allocated and reserved GPU memory."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")
    
    
class PPOBCTrainer(PPOTrainer):
    """
    PPO Trainer with Behavior Cloning (BC).

    This trainer extends the PPOTrainer to incorporate behavior cloning loss,
    leveraging expert data to guide the learning process.
    """
    def __init__(self, 
        bc_coef: float,
        mine_data: MineDataModule,
        callbacks: List[ObjectiveCallback],
        *args,
        **kwargs
    ):
        """
        Initializes the PPOBCTrainer.

        :param bc_coef: Coefficient for the behavior cloning loss.
        :type bc_coef: float
        :param mine_data: Data module for accessing expert data.
        :type mine_data: MineDataModule
        :param callbacks: List of callbacks for behavior cloning.
        :type callbacks: List[ObjectiveCallback]
        :param args: Positional arguments for the PPOTrainer.
        :param kwargs: Keyword arguments for the PPOTrainer.
        """
        super().__init__(*args, **kwargs)
        
        self.bc_coef = bc_coef
        self.data_module = mine_data
        self.callbacks = callbacks
        self.dataloader_iterator = iter(self.datamodule.train_dataloader())

        # self.kl_divergence_coef_rho = 0.0
        # self.coef_rho_decay = 1.0 # No decay for a coefficient of 0
        # self.ref_model = None
        # self.enable_ref_update = False # Disable ref model updates

    def train_iteration(self):
        """
        Performs a single training iteration, including PPO and BC updates.
        """

        gae_results = self.fetch_fragments_and_estimate_advantages(
            num_fragments=self.fragments_per_iteration,
        )
        self.ppo_update(
            records=gae_results["records"],
            td_targets=gae_results["td_targets"],
            advantages=gae_results["advantages"],
            old_logps=gae_results["old_logps"],
            old_vpreds=gae_results["old_vpreds"],
            rewards = gae_results["rewards"]
        )

        self.kl_divergence_coef_rho *= self.coef_rho_decay

    def ppo_update(self,
                  records: List[Tuple[FragmentIndex, str]],
                  td_targets: FragmentDataDict, 
                  advantages: FragmentDataDict,
                  old_logps: FragmentDataDict,
                  old_vpreds: FragmentDataDict,
                  rewards: FragmentDataDict
                  ):
        """
        Performs the PPO update step, incorporating BC loss.

        :param records: List of fragment records.
        :type records: List[Tuple[FragmentIndex, str]]
        :param td_targets: TD targets for value function update.
        :type td_targets: FragmentDataDict
        :param advantages: Advantages for policy update.
        :type advantages: FragmentDataDict
        :param old_logps: Log probabilities of actions under the old policy.
        :type old_logps: FragmentDataDict
        :param old_vpreds: Value predictions from the old policy.
        :type old_vpreds: FragmentDataDict
        :param rewards: Rewards received during rollouts.
        :type rewards: FragmentDataDict
        """
        
        self.buffer_reward = sum(rewards.values())
        if self.enable_ref_update:
            if self.buffer_reward>self.max_reward:
                self.max_reward = sum(rewards.values())
                self.ref_model = copy.deepcopy(self.inner_model)
                self.ref_model.eval()
                self.ref_version = self.num_updates
        mean_policy_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_kl_divergence_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_bc_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_entropy_bonus = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_value_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_total_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_approx_kl = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_clip_fraction = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_abs_td_target = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_abs_advantage = torchmetrics.MeanMetric().to(self.inner_model.device)
        explained_var_metric = torchmetrics.ExplainedVariance().to(self.inner_model.device)

        indexs = [index for index, _ in records]

        broken_num_lossnan = 0
        broken_num_kl = 0

        _advantage_sum1 = 0
        _advantage_sum2 = 0
        _advantage_count = 0
        for index in indexs:
            _advantage_sum1 += advantages[index].sum()
            _advantage_sum2 += (advantages[index] ** 2).sum()
            _advantage_count += np.prod(advantages[index].shape)
        advantage_mean = _advantage_sum1 / _advantage_count
        advantage_std = (_advantage_sum2 / _advantage_count - advantage_mean ** 2) ** 0.5
        torch.cuda.empty_cache()

        dataloader_iterator = iter(self.datamodule.train_dataloader())

        for epoch in range(self.epochs_per_iteration):

            it = data_iter(
                loader_pool=self.loader_pool,
                records=records,
                batch_size=self.batch_size_per_gpu,
                prefetch_batches=self.prefetch_batches
            )
            
            batch_count = 0
            if self.rank == 0:
                it = tqdm_ray.tqdm(it, desc=f"PPO update {self.num_updates + 1} at epoch {epoch + 1} / {self.epochs_per_iteration}", total=len(records) // self.batch_size_per_gpu)

            self.optimizer.zero_grad()
                
            for _batch in it:
                batch_fragments: List[SampleFragment] = _batch["fragment"] # type: ignore

                self.fragment_length = len(batch_fragments[0].next_done) # TODO: replace this with a better way

                # Prepare data
                batch = prepare_batch(self.inner_model, batch_fragments)
                
                B, T = batch["first"].shape #obs state action first
                #print("obs shape: ", batch["obs"]["img"].shape)
                batch_count += 1

                _old_logp = old_logps.format_batch(batch_fragments, device=self.inner_model.device)
                _advantage = advantages.format_batch(batch_fragments, device=self.inner_model.device)
                _old_vpred = old_vpreds.format_batch(batch_fragments, device=self.inner_model.device)
                _td_target = td_targets.format_batch(batch_fragments, device=self.inner_model.device)

                if self.normalize_advantage_full_batch:
                    _advantage = (_advantage - advantage_mean) / (advantage_std + 1e-8)

                new_state = batch["state"]
                if self.kl_divergence_coef_rho != 0:
                    assert self.ref_model is not None
                    new_ref_state = self.ref_model.initial_state(B)
                
                # Train

                if self.bc_coef != 0:
                    # calculate behavior cloning loss
                    try:
                        bc_batch = self.dataloader_iterator.next().to(self.inner_model.device)
                    except StopIteration:
                        self.dataloader_iterator = iter(self.datamodule.train_dataloader())
                        bc_batch = self.dataloader_iterator.next().to(self.inner_model.device)
                    memory_in = self.model.initial_state(bc_batch["image"].shape[0])
                    bc_latents, _ = self.model(bc_batch, memory_in)
                    bc_result = {'loss': 0}
                    for callback in self.callbacks:
                        call_result = callback(bc_batch, None, 'train', bc_latents, self.model)
                        for key, val in call_result.items():
                            bc_result[key] = bc_result.get(key, 0) + val
                    # rename loss to bc_loss
                    for key in bc_result.keys():
                        if "loss" in key:
                            bc_result["bc_loss"] = bc_result.pop(key)
                            break
                else:
                    bc_result = {'bc_loss': 0}

                first_backward = True
                for start in range(0, T, self.context_length):
                    end = min(T, start + self.context_length)
                    
                    #hack: This may need model-specific processing
                    chunk_obs = auto_slice(batch["obs"], start, end, dim=1, type_list=1)
                    chunk_first = auto_slice(batch["first"], start, end, dim=1, type_list=1)
                    chunk_action = auto_slice(batch["action"], start, end, dim=1, type_list=1)
    
                    old_logp = auto_slice(_old_logp, start, end, dim=1)
                    advantage: torch.Tensor = auto_slice(_advantage, start, end, dim=1) # type: ignore
                    old_vpred = auto_slice(_old_vpred, start, end, dim=1)
                    td_target = auto_slice(_td_target, start, end, dim=1)

                    loss_weight = (end - start) / T
                    
                    context = self.model.no_sync() if (isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and (batch_count % self.gradient_accumulation != 0 or end < T)) else nullcontext()
                    with context:
                        forward_result, new_state= self.model(input=chunk_obs, state_in=new_state, context = {"first": chunk_first})#, train_iter = str(self.num_updates))#, train_iter = uuid.uuid1().hex)#, train_iters = 2*self.num_optimized)
                        new_state = recursive_detach(new_state)
                        pi_logits = forward_result["pi_logits"]

                        if self.kl_divergence_coef_rho != 0:
                            with torch.inference_mode():#torch.inference_mode
                                ref_forward_result, new_ref_state = self.ref_model(input=chunk_obs, state_in=new_ref_state, context={"first":chunk_first})#, train_iter = str(self.num_updates))#, train_iter = uuid.uuid1().hex)#), train_iters = 2*self.num_optimized+1) # type: ignore
                                ref_pi_logit = ref_forward_result["pi_logits"]
                            epsilon = 1e-8
                            #print("pi_logits_sum_1", torch.exp(pi_logits['buttons']).sum(dim = -1))
                            kl_divergence_loss = self.inner_model.pi_head.kl_divergence({key: (ref_pi_logit[key]+epsilon) for key in ref_pi_logit}, {key:(pi_logits[key]+epsilon) for key in pi_logits}).mean()
                                                                                              # , pi_logits+epsilon).mean() # TODO: kl(p, q) or kl(q, p) ?
                            if kl_divergence_loss < -0.1:
                                ray.util.pdb.set_trace()
                        else:
                            kl_divergence_loss = torch.tensor(0.0, device=self.inner_model.device)
                        
                        new_logp = self.inner_model.pi_head.logprob(chunk_action, pi_logits) 
                        log_ratio = torch.clamp(new_logp - old_logp, max=self.log_ratio_range)
                        ratio = log_ratio.exp()

                        #patch of hkc
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        approx_kl_tensor = torch.tensor([approx_kl], device='cuda')
                        dist.all_reduce(approx_kl_tensor, op=dist.ReduceOp.MAX)  # 获取所有进程中的最大 approx_kl
                        if approx_kl_tensor.item() > 10:
                            broken_num_kl += 1
                            print("too high kl")
                            break

                        _policy_loss1 = - advantage * ratio
                        _policy_loss2 = - advantage * torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
                        policy_loss = torch.max(_policy_loss1, _policy_loss2).mean()

                        vpred = forward_result["vpred"].reshape(B, end - start)
                        #vpred = vpred.reshape(B, end-start)
                        
                        # TODO: should we halve the value loss?
                        if self.use_normalized_vf:
                            vf_loss_func = lambda vpred, td_target: (
                                0.5 * self.inner_model.value_head.loss(vpred, td_target, reduction="none") # type: ignore
                            )
                        else:
                            vf_loss_func = lambda vpred, td_target: (
                                0.5 * F.mse_loss(vpred, td_target, reduction="none")
                            )

                        vf_loss_BT = vf_loss_func(vpred, td_target)

                        if self.clip_vloss:
                            vpred_clipped = old_vpred + torch.clamp(
                                vpred - old_vpred,
                                -self.ppo_clip,
                                self.ppo_clip,
                            )
                            vf_loss_clipped_BT = vf_loss_func(vpred_clipped, td_target)
                            vf_loss_BT = torch.max(vf_loss_BT, vf_loss_clipped_BT)
                        
                        vf_loss = vf_loss_BT.mean()
                        
                        entropy_bonus = self.inner_model.pi_head.entropy(pi_logits).mean()

                        if self.num_updates < self.vf_warmup:
                            total_loss = (
                                self.ppo_vf_coef * vf_loss + 
                                1.0 * kl_divergence_loss + self.bc_coef * bc_result['bc_loss']
                            ) / self.gradient_accumulation
                        else:
                            total_loss = (
                                self.ppo_policy_coef * policy_loss + 
                                self.kl_divergence_coef_rho * kl_divergence_loss +
                                self.ppo_vf_coef * vf_loss + self.bc_coef * bc_result['bc_loss']
                                - self.entropy_bonus_coef * entropy_bonus
                            ) / self.gradient_accumulation

                        total_loss *= loss_weight

                        loss_tensor = torch.tensor([total_loss.item()], device='cuda')
                        is_nan = torch.isnan(loss_tensor).float()
                        dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)
                        if is_nan.item() > 0:
                            broken_num_lossnan += 1
                            print("loss nan")
                            break                            

                        total_loss.backward()

                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - log_ratio).mean()
                            if approx_kl > 100000:
                                ray.util.pdb.set_trace()
                            mean_approx_kl.update(approx_kl.detach(), weight=loss_weight)
                            clipfrac = ((ratio - 1.0).abs() > self.ppo_clip).float().mean()
                            mean_clip_fraction.update(clipfrac.detach(), weight=loss_weight)
                            mean_policy_loss.update(policy_loss.detach(), weight=loss_weight) # .detach() is necessary here
                            mean_kl_divergence_loss.update(kl_divergence_loss.detach(), weight=loss_weight)
                            mean_bc_loss.update(bc_result['bc_loss'].detach(), weight=loss_weight)
                            mean_value_loss.update(vf_loss.detach(), weight=loss_weight)
                            mean_entropy_bonus.update(entropy_bonus.detach(), weight=loss_weight)
                            mean_total_loss.update(total_loss.detach(), weight=loss_weight)

                            if self.use_normalized_vf:
                                vpred_denormalized = self.inner_model.value_head.denormalize(vpred).reshape(B, end - start) # type: ignore
                                explained_var_metric.update(vpred_denormalized.detach().reshape(-1), td_target.reshape(-1)) # TODO: weight?
                            else:
                                explained_var_metric.update(vpred.detach().reshape(-1), td_target.reshape(-1))

                            mean_abs_td_target.update(td_target.abs().mean().detach(), weight=loss_weight)
                            mean_abs_advantage.update(advantage.abs().mean().detach(), weight=loss_weight)

                if batch_count % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        mean_kl_divergence_loss_item = mean_kl_divergence_loss.compute().item()
        info = {
            "trainer/policy_loss": mean_policy_loss.compute().item(),
            "trainer/kl_divergence_loss": mean_kl_divergence_loss_item,
            "trainer/entropy_bonus": mean_entropy_bonus.compute().item(),
            "trainer/value_loss": mean_value_loss.compute().item(),
            "trainer/total_loss": mean_total_loss.compute().item(),
            "trainer/approx_kl": mean_approx_kl.compute().item(),
            "trainer/clip_fraction": mean_clip_fraction.compute().item(),
            "trainer/learning_rate": self.optimizer.param_groups[0]["lr"],
            "trainer/rho": self.kl_divergence_coef_rho,
            "trainer/explained_var": explained_var_metric.compute().item(), # type: ignore
            "trainer/abs_advantage": mean_abs_advantage.compute().item(),
            "trainer/abs_td_target": mean_abs_td_target.compute().item(),
            "trainer/ref_version": self.ref_version,
            "trainer/broken_num_lossnan": broken_num_lossnan,
            "trainer/broken_num_kl": broken_num_kl,
            "trainer/buffer_reward": self.buffer_reward,
            #"trainer/max_bonus": torch.max(torch.abs(self.inner_model.policy.net.zv_bonus)).item(),
        }
        for key, value in bc_result.items():
            info["trainer/" + key] = value

        self.num_updates += 1

        if self.rank == 0:
            if self.num_updates % self.save_interval == 0:
                # TODO: this may cause problem in distributed training
                logging.getLogger("ray").info(f"Saving checkpoint at update count {self.num_updates}...")
                
                if self.save_path:
                    checkpoint_dir = Path(self.save_path) / 'checkpoints' / self.time_stamp /str(self.num_updates)
                else:
                    checkpoint_dir = Path("checkpoints") / self.time_stamp /str(self.num_updates)

                logging.getLogger("ray").info(f"Checkpoint dir: {checkpoint_dir.absolute()}")
                if not checkpoint_dir.exists():
                    checkpoint_dir.mkdir(parents=True)

                #save model
                torch.save(self.inner_model.state_dict(), str(checkpoint_dir / "model.ckpt"))
                torch.save(self.optimizer.state_dict(), str(checkpoint_dir / "optimizer.ckpt"))
                with open(checkpoint_dir / "whole_config.py", "w") as f:
                    f.write(self.whole_config)

                if (
                    self.last_checkpoint_dir
                    and (self.num_updates + self.save_interval) % self.keep_interval != 0
                ):
                    shutil.rmtree(self.last_checkpoint_dir)
                self.last_checkpoint_dir = checkpoint_dir

            #send signal to record video
            SPS_all_workers = self.fragments_per_iteration * self.fragment_length / (time.time() - self.last_log_time)
            self.trained_steps_all_workers += self.fragments_per_iteration * self.fragment_length
            self.last_log_time = time.time()
            info["trainer/env_SPS_all_workers"] = SPS_all_workers
            info["trainer/env_steps_all_workers"] = self.trained_steps_all_workers
            print("I have send signal to manager: " + str(self.num_updates % self.record_video_interval == 0))
            ray.get(self.rollout_manager.log_statistics.remote(self.trained_steps_all_workers, self.num_updates % self.record_video_interval == 0))
            wandb_logger.log(info)
