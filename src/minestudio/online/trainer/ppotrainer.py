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
from minestudio.simulator import MinecraftSim
import time
import ray
import ray.train.torch
import os
import torchmetrics
import logging
from minestudio.online.trainer.basetrainer import BaseTrainer
from ray.experimental import tqdm_ray
from minestudio.online.utils import auto_stack
import uuid
import copy
import pickle
import torch.distributed as dist

VERBOSE = False

def print_memory_usage():
    """
    Print current CUDA memory usage information.
    
    This function displays the allocated and reserved memory on the current CUDA device
    in megabytes (MB). Useful for debugging memory issues during training.
    
    :returns: None (prints memory information to stdout)
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")
    
    
class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization (PPO) trainer for reinforcement learning.
    
    This class implements the PPO algorithm for training policies in Minecraft environments.
    It extends BaseTrainer and provides functionality for distributed training with Ray,
    gradient accumulation, value function warmup, and various PPO-specific optimizations
    including clipping, entropy bonuses, and KL divergence regularization.
    
    The trainer supports both policy and value function training with configurable
    coefficients, learning rate annealing, and checkpoint saving capabilities.
    """
    def __init__(self, 
        num_iterations: int,
        learning_rate: float,
        anneal_lr_linearly: bool,
        weight_decay: float,
        adam_eps: float,
        batch_size_per_gpu: int,
        batches_per_iteration: int,
        gradient_accumulation: int,
        epochs_per_iteration: int,
        vf_warmup: int,
        ppo_clip: float,
        clip_vloss: bool,
        max_grad_norm: float,
        zero_initial_vf: bool,
        ppo_vf_coef: float,
        ppo_policy_coef: float,
        kl_divergence_coef_rho: float,
        entropy_bonus_coef: float,
        coef_rho_decay: float,
        normalize_advantage_full_batch: bool,
        record_video_interval: int,
        save_interval: int,
        save_path: Optional[str],
        keep_interval: int,
        log_ratio_range: float,
        whole_config: str,
        enable_ref_update: bool = False,
        **kwargs
    ):
        """
        Initialize the PPO trainer with configuration parameters.
        
        :param num_iterations: Total number of training iterations to perform
        :param learning_rate: Initial learning rate for the optimizer
        :param anneal_lr_linearly: Whether to linearly anneal learning rate over training
        :param weight_decay: Weight decay coefficient for AdamW optimizer
        :param adam_eps: Epsilon parameter for AdamW optimizer numerical stability
        :param batch_size_per_gpu: Number of fragments processed per GPU in each batch
        :param batches_per_iteration: Number of batches processed per training iteration
        :param gradient_accumulation: Number of batches to accumulate gradients over
        :param epochs_per_iteration: Number of epochs to train on collected data per iteration
        :param vf_warmup: Number of iterations to train only value function before policy
        :param ppo_clip: Clipping parameter for PPO objective (typically 0.1-0.3)
        :param clip_vloss: Whether to clip value function loss using PPO clipping
        :param max_grad_norm: Maximum gradient norm for gradient clipping
        :param zero_initial_vf: Whether to initialize value function weights to zero
        :param ppo_vf_coef: Coefficient for value function loss in total loss
        :param ppo_policy_coef: Coefficient for policy loss in total loss  
        :param kl_divergence_coef_rho: Initial coefficient for KL divergence regularization
        :param entropy_bonus_coef: Coefficient for entropy bonus in policy loss
        :param coef_rho_decay: Decay factor for KL divergence coefficient per iteration
        :param normalize_advantage_full_batch: Whether to normalize advantages across full batch
        :param record_video_interval: Interval (in iterations) to record training videos
        :param save_interval: Interval (in iterations) to save model checkpoints
        :param save_path: Directory path to save checkpoints and logs
        :param keep_interval: Interval to keep checkpoints (others are deleted)
        :param log_ratio_range: Maximum allowed log probability ratio for stability
        :param whole_config: Complete configuration string for saving with checkpoints
        :param enable_ref_update: Whether to enable reference model updates
        :param kwargs: Additional arguments passed to parent BaseTrainer class
        """
        super().__init__(inference_batch_size_per_gpu=batch_size_per_gpu, **kwargs)
        
        wandb_logger.define_metric("trainer/*", step_metric="trainer/env_steps_all_workers")

        self.vf_warmup = vf_warmup
        self.num_iterations = num_iterations
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batches_per_iteration = batches_per_iteration
        self.epochs_per_iteration = epochs_per_iteration
        self.zero_initial_vf = zero_initial_vf
        self.ppo_clip = ppo_clip
        self.ppo_vf_coef = ppo_vf_coef
        self.kl_divergence_coef_rho = kl_divergence_coef_rho
        self.entropy_bonus_coef = entropy_bonus_coef
        self.coef_rho_decay = coef_rho_decay
        self.normalize_advantage_full_batch = normalize_advantage_full_batch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ppo_policy_coef = ppo_policy_coef
        self.adam_eps = adam_eps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.anneal_lr_linearly = anneal_lr_linearly
        self.clip_vloss = clip_vloss
        self.log_ratio_range = log_ratio_range
        self.fragments_per_iteration = self.num_workers * self.batch_size_per_gpu * self.batches_per_iteration
        self.record_video_interval = record_video_interval
        self.save_interval = save_interval
        self.keep_interval = keep_interval
        self.save_path = save_path
        self.enable_ref_update = enable_ref_update
        self.whole_config = whole_config
        assert self.batches_per_iteration % self.gradient_accumulation == 0

    def setup_model_and_optimizer(self, policy_generator) -> Tuple[MinePolicy, torch.optim.Optimizer]:
        """
        Set up the model and optimizer for PPO training.
        
        Creates the main policy model using the provided generator function and configures
        an AdamW optimizer with the specified hyperparameters. Optionally initializes
        value function weights to zero and sets up a reference model for KL divergence
        regularization if enabled.
        
        :param policy_generator: Function that returns a new instance of the policy model
        :returns: Tuple of (model, optimizer) ready for training
        :raises AssertionError: If reference model setup fails when KL regularization is enabled
        """
    
        model = policy_generator()
        if self.use_amp:
            scaler = torch.amp.GradScaler('cuda')
        else:
            scaler = None
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_eps
        )
        if self.zero_initial_vf:
            for param in model.value_head.parameters():
                param.data.zero_()
        logging.getLogger("ray").info(f"Model prepared. Type: {type(model)}")
        print("basic_config")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler("ray.log"),
                logging.StreamHandler()
            ]
        )
        model.train()

        if self.kl_divergence_coef_rho != 0:
            self.ref_model = self.policy_generator()
            self.ref_model.to(ray.train.torch.get_device())
            self.ref_model.train()
        else:
            self.ref_model = None
        return model, optimizer, scaler
    
    def train(self):
        """
        Execute the main PPO training loop.
        
        Runs the complete training process for the specified number of iterations.
        Each iteration involves collecting rollout data, computing advantages, and
        performing PPO updates. Handles learning rate annealing, model broadcasting
        to rollout workers, and checkpoint management.
        
        The method supports resuming from checkpoints by detecting current learning
        rate and calculating the appropriate starting iteration. Includes distributed
        training coordination and logging.
        
        :returns: None
        """
        self.num_updates = 0
        self.max_reward = 0
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print("Begining training....")

        if self.rank == 0:
            self.last_log_time = time.time()
            self.trained_steps_all_workers = 0
            self.last_checkpoint_dir = None

        current_lr = self.optimizer.param_groups[0]["lr"]
        #patch of hkc
        if current_lr > self.learning_rate:
            current_lr = self.learning_rate
        if self.learning_rate >0.000000001:
            self.num_updates =int((1.0 - current_lr / self.learning_rate)*self.num_iterations+0.00001)
        else:
            self.num_updates = 0
        self.kl_divergence_coef_rho = self.kl_divergence_coef_rho * (self.coef_rho_decay ** self.num_updates)

        for i in range(self.num_updates, self.num_iterations):
            print(f"[num_iters]: {i}")
            if self.anneal_lr_linearly:
                frac = 1.0 - i / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            self.train_iteration()
            if self.rank == 0:
                start_time = time.time()
                if self.num_updates > self.vf_warmup:
                    self.broadcast_model_to_rollout_workers(new_version=True)
                end_time = time.time()
                logging.getLogger("ray").info(f"Updated model in {end_time - start_time} seconds.")

    def train_iteration(self):
        """
        Execute a single training iteration of the PPO algorithm.
        
        Performs one complete iteration consisting of:
        1. Fetching trajectory fragments from rollout workers
        2. Computing Generalized Advantage Estimation (GAE) 
        3. Performing PPO updates on the collected data
        4. Decaying the KL divergence coefficient
        
        This method coordinates between data collection and policy optimization phases
        of the PPO algorithm.
        
        :returns: None
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
        Perform PPO policy and value function updates on collected trajectory data.
        
        Implements the core PPO algorithm including:
        - Policy loss computation with probability ratio clipping
        - Value function loss with optional clipping
        - KL divergence regularization against reference policy
        - Entropy bonus for exploration
        - Advantage normalization and gradient accumulation
        - Distributed training synchronization and error handling
        
        The method processes data in batches across multiple epochs, computing various
        metrics and losses while handling numerical stability issues. Includes 
        checkpoint saving and performance logging.
        
        :param records: List of (fragment_index, worker_id) tuples identifying data fragments
        :param td_targets: Temporal difference targets for value function training
        :param advantages: Computed advantages for policy gradient estimation
        :param old_logps: Log probabilities from the policy that collected the data
        :param old_vpreds: Value predictions from the policy that collected the data  
        :param rewards: Reward values from the trajectory fragments
        :returns: None
        :raises AssertionError: If reference model is None when KL regularization is enabled
        """
        
        self.buffer_reward = sum(rewards.values()) / len(rewards)
        mean_policy_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_kl_divergence_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_entropy_bonus = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_value_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_total_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_approx_kl = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_clip_fraction = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_abs_td_target = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_abs_advantage = torchmetrics.MeanMetric().to(self.inner_model.device)
        explained_var_metric = torchmetrics.ExplainedVariance().to(self.inner_model.device)

        if VERBOSE:
            debug_metrics = {}
            debug_metric_names = [
                "advantage_mean", "advantage_max", "advantage_min", "advantage_std",
                "ratio_mean", "ratio_max", "ratio_min", "ratio_std",
                "entropy_mean", "entropy_max", "entropy_min", "entropy_std",
                "vf_pred_mean", "vf_pred_max", "vf_pred_min", "vf_pred_std",
                "log_p_mean", "log_p_max", "log_p_min", "log_p_std", "old_log_p_mean", "old_log_p_max", "old_log_p_min", "old_log_p_std"
            ]

            for name in debug_metric_names:
                debug_metrics[name] = torchmetrics.MeanMetric().to(self.inner_model.device)

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
                    with context, torch.cuda.amp.autocast(enabled=self.use_amp):
                        forward_result, new_state= self.model(input=chunk_obs, state_in=new_state, context = {"first": chunk_first})#, train_iter = str(self.num_updates))#, train_iter = uuid.uuid1().hex)#, train_iters = 2*self.num_optimized)
                        new_state = recursive_detach(new_state)
                        pi_logits = forward_result["pi_logits"]

                        if self.kl_divergence_coef_rho != 0:
                            with torch.inference_mode():#torch.inference_mode
                                ref_forward_result, new_ref_state = self.ref_model(input=chunk_obs, state_in=new_ref_state, context={"first":chunk_first})#, train_iter = str(self.num_updates))#, train_iter = uuid.uuid1().hex)#), train_iters = 2*self.num_optimized+1) # type: ignore
                                ref_pi_logit = ref_forward_result["pi_logits"]
                            epsilon = 1e-8
                            #print("pi_logits_sum_1", torch.exp(pi_logits['buttons']).sum(dim = -1))
                            # kl_divergence_loss = self.inner_model.pi_head.kl_divergence({key: (ref_pi_logit[key]+epsilon) for key in ref_pi_logit}, {key:(pi_logits[key]+epsilon) for key in pi_logits}).mean() # TODO: kl(p, q) or kl(q, p) ?
                            kl_divergence_loss = self.inner_model.pi_head.kl_divergence({key:(pi_logits[key]+epsilon) for key in pi_logits}, {key: (ref_pi_logit[key]+epsilon) for key in ref_pi_logit}).mean() # TODO: kl(p, q) or kl(q, p) ?
                            if kl_divergence_loss < -0.1:
                                ray.util.pdb.set_trace()
                        else:
                            kl_divergence_loss = torch.tensor(0.0, device=self.inner_model.device)
                        
                        new_logp = self.inner_model.pi_head.logprob(chunk_action, pi_logits) 

                        # !FIX:Clamp new_logp where advantage is negative
                        condition = advantage < 0
                        new_logp = torch.where(condition, torch.clamp(new_logp, min=-11.0), new_logp)

                        log_ratio = torch.clamp(new_logp - old_logp, max=self.log_ratio_range)
                        ratio = log_ratio.exp()

                        #patch of hkc
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        approx_kl_tensor = torch.tensor([approx_kl], device='cuda')
                        dist.all_reduce(approx_kl_tensor, op=dist.ReduceOp.MAX)  # 获取所有进程中的最大 approx_kl
                        if approx_kl_tensor.item() > 10:
                            broken_num_kl += 1
                            print("too high kl")
                            # break

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
                                1.0 * kl_divergence_loss
                            ) / self.gradient_accumulation
                        else:
                            total_loss = (
                                self.ppo_policy_coef * policy_loss + 
                                self.kl_divergence_coef_rho * kl_divergence_loss +
                                self.ppo_vf_coef * vf_loss
                                - self.entropy_bonus_coef * entropy_bonus
                            ) / self.gradient_accumulation

                        total_loss *= loss_weight
                        # assert not torch.isnan(total_loss)

                        # 假设 loss 是你的损失值
                        loss_tensor = torch.tensor([total_loss.item()], device='cuda')
                        is_nan = torch.isnan(loss_tensor).float()
                        dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)
                        if is_nan.item() > 0:
                            broken_num_lossnan += 1
                            print("loss nan")
                            break

                        if self.use_amp:
                            self.scaler.scale(total_loss).backward()
                        else:
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
                            mean_value_loss.update(vf_loss.detach(), weight=loss_weight)
                            mean_entropy_bonus.update(entropy_bonus.detach(), weight=loss_weight)
                            mean_total_loss.update(total_loss.detach(), weight=loss_weight)

                            if VERBOSE:
                                current_entropy = self.inner_model.pi_head.entropy(pi_logits)
                                current_vpred = forward_result["vpred"]
                                
                                debug_metrics["advantage_mean"].update(advantage.mean().detach(), weight=loss_weight)
                                debug_metrics["advantage_max"].update(advantage.max().detach(), weight=loss_weight)
                                debug_metrics["advantage_min"].update(advantage.min().detach(), weight=loss_weight)
                                debug_metrics["advantage_std"].update(advantage.std().detach(), weight=loss_weight)
                                
                                debug_metrics["ratio_mean"].update(ratio.mean().detach(), weight=loss_weight)
                                debug_metrics["ratio_max"].update(ratio.max().detach(), weight=loss_weight)
                                debug_metrics["ratio_min"].update(ratio.min().detach(), weight=loss_weight)
                                debug_metrics["ratio_std"].update(ratio.std().detach(), weight=loss_weight)

                                debug_metrics["entropy_mean"].update(current_entropy.mean().detach(), weight=loss_weight)
                                debug_metrics["entropy_max"].update(current_entropy.max().detach(), weight=loss_weight)
                                debug_metrics["entropy_min"].update(current_entropy.min().detach(), weight=loss_weight)
                                debug_metrics["entropy_std"].update(current_entropy.std().detach(), weight=loss_weight)

                                debug_metrics["vf_pred_mean"].update(current_vpred.mean().detach(), weight=loss_weight)
                                debug_metrics["vf_pred_max"].update(current_vpred.max().detach(), weight=loss_weight)
                                debug_metrics["vf_pred_min"].update(current_vpred.min().detach(), weight=loss_weight)
                                debug_metrics["vf_pred_std"].update(current_vpred.std().detach(), weight=loss_weight)

                                debug_metrics["log_p_mean"].update(new_logp.mean().detach(), weight=loss_weight)
                                debug_metrics["log_p_max"].update(new_logp.max().detach(), weight=loss_weight)
                                debug_metrics["log_p_min"].update(new_logp.min().detach(), weight=loss_weight)
                                debug_metrics["log_p_std"].update(new_logp.std().detach(), weight=loss_weight)

                                debug_metrics["old_log_p_mean"].update(old_logp.mean().detach(), weight=loss_weight)
                                debug_metrics["old_log_p_max"].update(old_logp.max().detach(), weight=loss_weight)
                                debug_metrics["old_log_p_min"].update(old_logp.min().detach(), weight=loss_weight)
                                debug_metrics["old_log_p_std"].update(old_logp.std().detach(), weight=loss_weight)

                            if self.use_normalized_vf:
                                vpred_denormalized = self.inner_model.value_head.denormalize(vpred).reshape(B, end - start) # type: ignore
                                explained_var_metric.update(vpred_denormalized.detach().reshape(-1), td_target.reshape(-1)) # TODO: weight?
                            else:
                                explained_var_metric.update(vpred.detach().reshape(-1), td_target.reshape(-1))

                            mean_abs_td_target.update(td_target.abs().mean().detach(), weight=loss_weight)
                            mean_abs_advantage.update(advantage.abs().mean().detach(), weight=loss_weight)

                if batch_count % self.gradient_accumulation == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
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
            # "trainer/ref_version": self.ref_version,
            "trainer/broken_num_lossnan": broken_num_lossnan,
            "trainer/broken_num_kl": broken_num_kl,
            "trainer/buffer_reward": self.buffer_reward,
            #"trainer/max_bonus": torch.max(torch.abs(self.inner_model.policy.net.zv_bonus)).item(),
        }

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
                print(f"Saving model to {checkpoint_dir / 'model.ckpt'}")
                torch.save(self.inner_model.state_dict(), str(checkpoint_dir / "model.ckpt"))
                torch.save(self.optimizer.state_dict(), str(checkpoint_dir / "optimizer.ckpt"))
                if self.use_amp:
                    torch.save(self.scaler.state_dict(), str(checkpoint_dir / "scaler.ckpt"))
                with open(checkpoint_dir / "whole_config.pkl", "wb") as f:
                    pickle.dump(self.whole_config, f)

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
            