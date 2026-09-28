from functools import partial
import uuid
import time
import torch
import torch.multiprocessing as mp
import ray
from multiprocessing.connection import Connection
import rich
import logging
from minestudio.online.rollout.env_worker import EnvWorker
from typing import Optional, List, Protocol, Tuple, Any, Callable, Dict
import numpy as np
import random
from minestudio.models import MinePolicy
from minestudio.simulator import MinecraftSim
from ray.actor import ActorHandle
from minestudio.online.utils.rollout.monitor import PipelineMonitor, MovingStat
from minestudio.online.utils import auto_stack, auto_to_torch, auto_to_numpy
from minestudio.online.utils.rollout.datatypes import StepRecord

class ProgressHandler(Protocol):
    """
    A protocol defining the structure for a progress handler function.

    This handler is called by the RolloutWorker to report step-wise progress.
    Implementers of this protocol can use this to, for example, send data to a replay buffer.
    """
    def __call__(self, *,
        worker_uuid: str,
        obs: Dict[str, Any],
        state: List[torch.Tensor],
        action: Dict[str, Any],
        last_reward: float,
        last_terminated: bool,
        last_truncated: bool,
        episode_uuid: str,
    ) -> None: ...

class RolloutWorker():
    """
    Manages a set of environments and a policy to collect experience for reinforcement learning.

    This class is responsible for:
    - Spawning and managing multiple environment worker processes (EnvWorker).
    - Performing inference using the policy model.
    - Stepping through the environments and collecting observations, actions, rewards, etc.
    - Communicating with an EpisodeStatistics actor to report episode-level metrics.
    - Optionally calling a progress_handler to process step-wise data (e.g., for a replay buffer).

    :param num_envs: The number of parallel environments to run.
    :param policy_generator: A callable that returns an instance of MinePolicy (the policy model).
    :param env_generator: A callable that returns an instance of MinecraftSim (the environment).
    :param use_normalized_vf: Whether to use a normalized value function. If True, vpreds will be denormalized.
    :param model_device: The device to run the policy model on (e.g., "cpu", "cuda:0").
    :param next_model_version: The initial version of the model.
    :param batch_size: The number of environment steps to batch together for inference.
    :param video_fps: Frames per second for video recording in EnvWorker.
    :param video_output_dir: Directory to save videos in EnvWorker.
    :param resume: Optional path to a checkpoint to resume training from.
    :param restart_interval: Optional interval in seconds after which to restart EnvWorkers.
    :param moving_stat_duration: Duration in seconds for calculating moving statistics (e.g., for pipeline monitoring).
    :param log_interval: Optional interval in seconds for logging pipeline monitoring stats.
    :param episode_statistics: Optional Ray ActorHandle for an EpisodeStatistics actor.
    :param progress_handler: Optional callable that conforms to the ProgressHandler protocol.
    :param max_fast_reset: Maximum number of fast resets for an EnvWorker before a full restart.
    :param rollout_worker_id: Identifier for this rollout worker.
    """
    def __init__(self, 
                 num_envs: int, 
                 policy_generator: Callable[[], MinePolicy],
                 env_generator: Callable[[], MinecraftSim],
                 use_normalized_vf: bool = False,
                 model_device: str = "cpu",
                 next_model_version: int = 0,
                 batch_size: int = 1,
                 video_fps: int = 20,
                 video_output_dir: str = "./output",
                 resume: Optional[str] = None,
                 restart_interval: Optional[int] = None,
                 moving_stat_duration: int = 300,
                 log_interval: Optional[int] = None, 
                 episode_statistics: Optional[ActorHandle] = None,
                 progress_handler: Optional[ProgressHandler] = None,
                 max_fast_reset: int = 10000,
                 rollout_worker_id: int  = 0): # TODO: more advanced batch strategy
        self.num_envs = num_envs
        self.policy_generator = policy_generator
        self.env_generator = env_generator
        self.model_device = model_device
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.progress_handler = progress_handler
        self.episode_statistics = episode_statistics
        self.use_normalized_vf = use_normalized_vf
        self.next_model_version = next_model_version
        self.num_resets = [0 for _ in range(self.num_envs)]
        self.env_conns: List[Connection] = []
        self.env_processes = []
        self.queued_requests: List[Tuple[int, Any]] = []
        self.agent_states = {}
        self.worker_uuids = [
            str(uuid.uuid4())
            for _ in range(self.num_envs)
        ]

        self.pipeline_monitor = PipelineMonitor([
            'recv_obs',
            'inference',
            'send_action',
        ], duration=moving_stat_duration)

        self.queue_size_after_inference_stat = MovingStat(
            duration=moving_stat_duration
        )

        self.video_step = 0

        if mp.get_start_method(allow_none=True) != "spawn":
            rich.print("[red]WARNING: The multiprocessing start method is not set to 'spawn'. We will set it to 'spawn' for you, but this may cause problems if you are using other libraries that use multiprocessing.[/red]")

        mp.set_start_method("spawn", force=True)

        self.progress_handler_kwargs: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        for env_id in range(self.num_envs):
            parent_conn, child_conn = mp.Pipe()
            self.env_conns.append(parent_conn) # type: ignore
            process = EnvWorker(env_generator=env_generator, conn=child_conn, restart_interval=restart_interval, video_fps=video_fps, video_output_dir=video_output_dir, max_fast_reset=max_fast_reset, env_id = env_id, rollout_worker_id = rollout_worker_id) # type: ignore
            self.env_processes.append(process)
            process.start()

        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = False # type: ignore
        # the input size of the model is not fixed, so cudnn.benchmark must be disabled

        self.agent: MinePolicy = policy_generator()
        if resume is not None:
            self.load_weights(torch.load(resume+"/model.ckpt"))
        self.agent.eval()
        self.agent.to(self.model_device)

        self.last_log_time = time.time()

    def update_model_version(self, next_model_version: int):
        """
        Updates the model version number that will be associated with subsequently collected data.

        :param next_model_version: The new model version number.
        """
        self.next_model_version = next_model_version

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        Loads new weights into the policy model.

        :param weights: A dictionary containing the state dictionary of the model.
        """
        #weights = weights.copy()
        for key, val in weights.items():
            weights[key] = val.to(self.model_device)
        self.agent.load_state_dict(weights, strict = False)
        self.agent.eval()
        torch.cuda.empty_cache()
        return

    def inference(self, requests: list) -> Tuple[list, list, list]:
        """
        Performs a batch of inference requests using the policy model.

        :param requests: A list of tuples, where each tuple contains (worker_id, observation).
        :returns: A tuple containing:
            - result_actions: A list of actions for each request.
            - result_states: A list of next hidden states for each request.
            - result_vpreds: A list of predicted values (vpreds) for each request.
        """
        # logger.info(f"requests: {requests}")
        worker_ids, inputs = zip(*requests)
        idds = []
        in_states = []

        for idx in worker_ids:
            self.pipeline_monitor.report_enter('inference', pipeline_id=self.worker_uuids[idx])
            idds.append(str(idx)+"_"+str(self.num_resets[idx]))
            in_states.append(self.agent_states[idx])
        
        # batch = self.agent.merge_input(inputs) #auto_to_torch(auto_stack([auto_stack([input]) for input in inputs]), device=self.model_device)
        batch = auto_to_torch(auto_stack([auto_stack([input]) for input in inputs]), device=self.model_device)
        memory_in = self.agent.merge_state(in_states)
        action = None

        with torch.inference_mode():
            latents, state_out = self.agent(input=batch, state_in=memory_in)
            pi_logits, vpreds = latents["pi_logits"], latents['vpred']
            action = self.agent.pi_head.sample(pi_logits, deterministic=False)
            if self.use_normalized_vf:
                vpreds = self.agent.value_head.denormalize(vpreds) # type: ignore

        result_actions = self.agent.split_action(action, len(worker_ids))
        result_states = self.agent.split_state(state_out, len(worker_ids))
        result_vpreds = [vpred[0].cpu().numpy() for vpred in vpreds]

        return result_actions, result_states, result_vpreds

    def poll_environments(self):
        """
        Polls all environment connections for messages and processes them.

        Handles different message types:
        - "step_agent": An environment is ready for a new action. The observation is added to queued_requests.
        - "reset_state": An environment has reset. The agent's hidden state for this environment is reset.
        - "report_rewards": An environment has finished an episode. Rewards are reported to EpisodeStatistics.
        """
        for i, conn in enumerate(self.env_conns):
            if conn.poll():
                args = conn.recv()
                if args[0] == "step_agent":
                    self.pipeline_monitor.report_enter('recv_obs', pipeline_id=self.worker_uuids[i])
                    obs, last_reward, last_terminated, last_truncated, episode_uuid = args[1], args[2], args[3], args[4], args[5]
                    self.progress_handler_kwargs[i] = {
                        "worker_uuid": self.worker_uuids[i],
                        "obs": obs,
                        "state": self.agent_states[i],
                        "last_reward": last_reward,
                        "last_terminated": last_terminated,
                        "last_truncated": last_truncated,
                        "episode_uuid": episode_uuid,
                    }
                    self.queued_requests.append((i, obs))
                elif args[0] == "reset_state":
                    self.agent_states[i] = self.agent.initial_state(1)
                    self.num_resets[i] += 1
                    conn.send("ok")
                elif args[0] == "report_rewards":
                    rewards = args[1]
                    task = args[2] if len(args) > 2 else None
                    if self.episode_statistics is not None:
                        video_step, episode_info = ray.get(self.episode_statistics.report_episode.remote(rewards, its_specfg = task if task is not None else ""))
                        if video_step is not None and video_step > self.video_step:
                            self.video_step = video_step
                            conn.send((video_step, episode_info))
                        else:    
                            conn.send((None, episode_info))
                    else:
                        conn.send((None, None))
                else:
                    raise NotImplementedError
                
    def loop(self) -> None:
        """
        Runs one iteration of the rollout loop.

        This involves:
        1. Polling environments until enough requests are queued to fill a batch.
        2. Performing inference on the batch of requests.
        3. Sending actions back to the environments.
        4. Calling the progress_handler if it's set.
        5. Polling environments again to process any immediate responses.
        6. Logging statistics if the log_interval is met.
        """
        while len(self.queued_requests) < self.batch_size:
            self.poll_environments()

        requests = self.queued_requests[:self.batch_size]
        actions, states, denormalized_vpreds = self.inference(requests)
        self.queued_requests = self.queued_requests[self.batch_size:]
        for_idx =  0
        for action, state, denormalized_vpred, (idx, _) in zip(actions, states, denormalized_vpreds, requests):
            self.pipeline_monitor.report_enter('send_action', pipeline_id=self.worker_uuids[idx])
            self.agent_states[idx] = state
            # logger.info(f"sending action: {action}")
            action = {k: v.reshape(-1) for k, v in action.items()}
            self.env_conns[idx].send((action, denormalized_vpred))
            if self.progress_handler is not None:
                self.progress_handler(
                    action=action,
                    **self.progress_handler_kwargs[idx],    
                )
            for_idx += 1

        self.poll_environments()
        self.queue_size_after_inference_stat.update(len(self.queued_requests))

        if self.log_interval is not None and time.time() - self.last_log_time > self.log_interval:
            self.last_log_time = time.time()
            self.pipeline_monitor.print()
            rich.print(f"Average queue size after inference: {self.queue_size_after_inference_stat.average()}")
    
    def rollout(self, num_batches: int) -> None:
        """
        Runs the rollout loop for a specified number of batches.

        :param num_batches: The number of inference batches to collect.
        """
        num_batches_left = num_batches
        while num_batches_left >= 0:
            self.loop()
            num_batches_left -= 1

    def progress_handler(self, *,
        worker_uuid: str,
        obs: Dict[str, Any],
        env_spec: str,
        state: List[torch.Tensor],
        action: Dict[str, Any],
        last_reward: float,
        last_terminated: bool,
        last_truncated: bool,
        episode_uuid: str
    ) -> None:
        """
        Default progress handler, likely intended to be overridden or replaced.
        This implementation seems to be a duplicate of logic within RolloutWorkerWrapper.progress_handler
        and might not be used if a custom progress_handler is provided during RolloutWorker initialization.

        :param worker_uuid: The UUID of the worker environment.
        :param obs: The observation from the environment.
        :param env_spec: Specification of the environment, used for filtering.
        :param state: The hidden state of the policy model.
        :param action: The action taken by the policy.
        :param last_reward: The reward received from the previous step.
        :param last_terminated: Whether the episode terminated at the previous step.
        :param last_truncated: Whether the episode was truncated at the previous step.
        :param episode_uuid: The UUID of the current episode.
        :raises AssertionError: if env_spec is not in self.env_spec.config4test when not in self.env_spec.config.
        """
        if env_spec in self.env_spec.config:
            if (
                len(self.buffer[worker_uuid]) > 0
                and (
                    self.current_model_version - self.buffer[worker_uuid][-1].model_version > self.max_staleness
                    or
                    self.current_session_id != self.buffer[worker_uuid][-1].session_id
                )
            ):
                self.buffer[worker_uuid] = []
                
            self.buffer[worker_uuid].append(StepRecord(
                worker_uuid=worker_uuid,
                obs=obs,
                state=None,
                action=action,
                last_reward=last_reward,
                last_terminated=last_terminated,
                last_truncated=last_truncated,
                model_version=self.current_model_version,
                episode_uuid=episode_uuid,
                session_id=self.current_session_id
            ))

            steps_to_send = None
            if len(self.buffer[worker_uuid]) > self.fragment_length:
                assert len(self.buffer[worker_uuid]) == self.fragment_length + 1
                steps_to_send = self.buffer[worker_uuid][:self.fragment_length]
                self.buffer[worker_uuid] = self.buffer[worker_uuid][self.fragment_length:]
                self.to_send_queue.put((steps_to_send, self.buffer[worker_uuid][0]))

            if len(self.buffer[worker_uuid]) == 1:
                self.buffer[worker_uuid][0].state = auto_to_numpy(state) #[s.cpu().numpy() for s in state]
        else :
            print("env_spec not in self.env_spec.config, do not send to buffer")
            assert(env_spec in self.env_spec.config4test)

if __name__ == "__main__":
    ray.init(address="auto", ignore_reinit_error=True, namespace="online",runtime_env={"env_vars": {"RAY_DEBUG": "1"}, })
    mp.set_start_method("spawn")
    # from minestudio.simulator.callbacks import (
    #     SpeedTestCallback, 
    #     RecordCallback, 
    #     SummonMobsCallback, 
    #     MaskActionsCallback, 
    #     RewardsCallback, 
    #     CommandsCallback, 
    #     TaskCallback,
    #     FastResetCallback
    # )

    # def env_generator():
    #     from minestudio.simulator import MinecraftSim
    #     from minestudio.simulator.callbacks import (
    #         SpeedTestCallback, 
    #         RecordCallback, 
    #         SummonMobsCallback, 
    #         MaskActionsCallback, 
    #         RewardsCallback, 
    #         CommandsCallback, 
    #         TaskCallback,
    #         FastResetCallback
    #     )
    #     sim = MinecraftSim(
    #         action_type="env",
    #         callbacks=[
    #             SpeedTestCallback(50), 
    #             SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
    #             MaskActionsCallback(inventory=0, camera=np.array([0., 0.])), 
    #             RecordCallback(record_path="./output", fps=30),
    #             RewardsCallback([{
    #                 'event': 'kill_entity', 
    #                 'objects': ['cow', 'sheep'], 
    #                 'reward': 1.0, 
    #                 'identity': 'kill sheep or cow', 
    #                 'max_reward_times': 5, 
    #             }]),
    #             CommandsCallback(commands=[
    #                 '/give @p minecraft:iron_sword 1',
    #                 '/give @p minecraft:diamond 64',
    #             ]), 
    #             FastResetCallback(
    #                 biomes=['mountains'],
    #                 random_tp_range=1000,
    #             ), 
    #             TaskCallback([
    #                 {'name': 'chop', 'text': 'mine the oak logs'}, 
    #                 {'name': 'diamond', 'text': 'mine the diamond ore'},
    #             ])
    #         ]
    #     )
    #     return sim

    # def policy_generator():
    #     from minestudio.models.openai_vpt.body import load_vpt_policy
    #     model_path = '/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model'
    #     weights_path = '/nfs-shared/jarvisbase/pretrained/rl-from-early-game-2x.weights'
    #     policy = load_vpt_policy(model_path, weights_path)
    #     return policy

    # worker = RolloutWorker(
    #     num_envs=8,
    #     env_generator=env_generator,
    #     policy_generator=policy_generator,
    #     model_device="cuda:4",
    #     batch_size=4,
    #     log_interval=5
    # )
    
    # worker.rollout(num_batches=1000000)