from queue import Queue
import ray
import numpy as np
from ray.actor import ActorHandle
from omegaconf import DictConfig
from minestudio.online.rollout.rollout_worker import RolloutWorker
from minestudio.online.utils import auto_stack, auto_to_numpy
from minestudio.online.utils.rollout.datatypes import SampleFragment
from minestudio.online.rollout.replay_buffer import ReplayBufferInterface
from minestudio.online.rollout.episode_statistics import EpisodeStatistics
from minestudio.online.utils.rollout.datatypes import FragmentMetadata, StepRecord
from typing import Dict, Union, Any, List, Optional, Callable
import torch
from dataclasses import dataclass
from collections import defaultdict
from minestudio.models import MinePolicy
from minestudio.simulator import MinecraftSim

WRAPPER_CONCURRENCY = 3
@ray.remote(max_concurrency=WRAPPER_CONCURRENCY) # type: ignore
class RolloutWorkerWrapper:
    """
    A Ray actor that wraps a RolloutWorker to manage its lifecycle and communication.

    :param fragment_length: The length of the fragments to be collected from the RolloutWorker.
    :param policy_generator: A callable that generates a policy model.
    :param env_generator: A callable that generates a Minecraft simulation environment.
    :param worker_config: A DictConfig object containing configuration for the RolloutWorker.
    :param episode_statistics: An ActorHandle for the EpisodeStatistics actor.
    :param to_send_queue_size: The size of the queue for sending fragments to the replay buffer.
    :param use_normalized_vf: A boolean indicating whether to use a normalized value function.
    :param resume: An optional string specifying a checkpoint to resume from.
    :param max_staleness: The maximum staleness allowed for model versions.
    :param rollout_worker_id: The ID of the rollout worker.
    """
    def __init__(
            self,
            fragment_length: int,
            policy_generator: Callable[[], MinePolicy],
            env_generator: Callable[[], MinecraftSim],
            worker_config: DictConfig,
            episode_statistics: ActorHandle,
            to_send_queue_size: int,
            use_normalized_vf: bool,
            resume: Optional[str],
            max_staleness: int,
            rollout_worker_id: int,
    ):

        self.max_staleness = max_staleness
        self.fragment_length = fragment_length
        self.use_normalized_vf = use_normalized_vf
        self.next_model_version = -1
        self.rollout_worker = RolloutWorker(
            model_device="cuda",
            progress_handler=self.progress_handler,
            episode_statistics=episode_statistics,
            use_normalized_vf=self.use_normalized_vf,
            resume=resume,
            next_model_version = self.next_model_version,
            rollout_worker_id = rollout_worker_id,
            policy_generator=policy_generator,
            env_generator=env_generator,
            **worker_config
        )
        self.buffer: Dict[str, List[StepRecord]] = defaultdict(list)
        self.current_model_version = -1
        self.current_session_id = ""
        self.next_session_id = ""
        self.next_model_version = -1
        self.next_state_dict = None
        self.num_fragments = defaultdict(int)
        self.to_send_queue = Queue(to_send_queue_size)
        self.replay_buffer = ReplayBufferInterface()

    def update_model(self, session_id: str, model_version: int, packed_state_dict_ref: List[ray.ObjectRef]):
        """
        Updates the model used by the RolloutWorker.

        :param session_id: The ID of the current training session.
        :param model_version: The version of the model to update to.
        :param packed_state_dict_ref: A list containing a Ray ObjectRef to the packed state dictionary of the model.
        :raises AssertionError: if packed_state_dict_ref does not contain exactly one element.
        """
        assert len(packed_state_dict_ref) == 1
        self.next_state_dict = ray.get(packed_state_dict_ref[0])
        self.next_model_version = model_version
        self.next_session_id = session_id
        self.rollout_worker.update_model_version(self.next_model_version) 

    def progress_handler(self, *,
        worker_uuid: str,
        obs: Dict[str, Any],
        state: List[torch.Tensor],
        action: Dict[str, Any],
        last_reward: float,
        last_terminated: bool,
        last_truncated: bool,
        episode_uuid: str
    ) -> None:
        """
        Handles the progress of the RolloutWorker, collecting steps and sending fragments.

        This method is called by the RolloutWorker after each step in the environment.
        It buffers the steps and, when enough steps are collected (fragment_length + 1),
        it creates a SampleFragment and puts it into a queue to be sent to the replay buffer.

        :param worker_uuid: The UUID of the worker environment.
        :param obs: The observation from the environment.
        :param state: The hidden state of the policy model.
        :param action: The action taken by the policy.
        :param last_reward: The reward received from the previous step.
        :param last_terminated: Whether the episode terminated at the previous step.
        :param last_truncated: Whether the episode was truncated at the previous step.
        :param episode_uuid: The UUID of the current episode.
        """
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


    def _send(self, steps: List[StepRecord], next_step: StepRecord):
        """
        Creates a SampleFragment from a list of steps and sends it to the replay buffer.

        :param steps: A list of StepRecord objects representing the steps in the fragment.
        :param next_step: The StepRecord object for the step immediately following the fragment.
        :raises AssertionError: if the number of rewards in the fragment is not equal to fragment_length.
        """
        last_next_done = next_step.last_terminated or next_step.last_truncated
        rewards = [step.last_reward for step in steps][1:] + [next_step.last_reward]
        next_done = [step.last_terminated or step.last_truncated for step in steps][1:] + [last_next_done]
        assert steps[0].state is not None

        fragment = SampleFragment(
            obs=auto_stack([step.obs for step in steps]),
            action=auto_stack([step.action for step in steps]),
            next_done=np.array(next_done, dtype=np.bool_),
            reward=np.array(rewards, dtype=np.float32),
            first=np.array([step.last_terminated or step.last_truncated for step in steps], dtype=np.bool_),
            episode_uuids=[step.episode_uuid for step in steps],
            in_state=steps[0].state,
            worker_uuid=steps[0].worker_uuid,
            fid_in_worker=self.num_fragments[steps[0].worker_uuid],
            next_obs=next_step.obs,
        )

        self.num_fragments[fragment.worker_uuid] += 1

        assert fragment.reward.shape[0] == self.fragment_length

        model_version = steps[0].model_version
        session_id = steps[0].session_id

        self.replay_buffer.add_fragment(
            fragment=fragment,
            metadata=FragmentMetadata(
                session_id=session_id,
                model_version=model_version,
                worker_uuid=fragment.worker_uuid,
                fid_in_worker=fragment.fid_in_worker
            )
        )
            
    def rollout_thread(self):
        """
        The main thread for the rollout process.
        Continuously runs the RolloutWorker's loop and updates the model when a new one is available.
        """
        while True:
            if self.next_state_dict is not None:
                self.rollout_worker.load_weights(self.next_state_dict)
                self.current_model_version = self.next_model_version
                self.current_session_id = self.next_session_id
                self.next_state_dict = None
            self.rollout_worker.loop()

    def sender_thread(self):
        """
        The main thread for sending collected fragments to the replay buffer.
        Continuously gets fragments from the to_send_queue and calls _send.
        """
        while True:
            steps_to_send, next_step = self.to_send_queue.get()
            self._send(steps_to_send, next_step)

class _RolloutManager:
    """
    Manages a collection of RolloutWorkerWrappers to perform distributed rollouts.

    This class is responsible for creating, starting, and managing multiple
    RolloutWorkerWrapper actors. It handles model updates for these workers
    and interfaces with the ReplayBuffer and EpisodeStatistics actors.

    :param policy_generator: A callable that generates a policy model.
    :param env_generator: A callable that generates a Minecraft simulation environment.
    :param use_normalized_vf: A boolean indicating whether to use a normalized value function.
    :param discount: The discount factor for calculating discounted rewards.
    :param num_rollout_workers: The number of rollout workers to create.
    :param num_cpus_per_worker: The number of CPUs to allocate for each rollout worker.
    :param num_gpus_per_worker: The number of GPUs to allocate for each rollout worker.
    :param to_send_queue_size: The size of the queue for sending fragments to the replay buffer in each worker.
    :param fragment_length: The number of steps in each fragment.
    :param resume: An optional string specifying a checkpoint to resume from.
    :param replay_buffer_config: A DictConfig object containing configuration for the replay buffer.
    :param worker_config: A DictConfig object containing configuration for the rollout workers.
    :param episode_statistics_config: A DictConfig object containing configuration for episode statistics.
    """
    def __init__(
            self,
            policy_generator: Callable[[], MinePolicy],
            env_generator: Callable[[], MinecraftSim],
            use_normalized_vf: bool,
            discount: float,
            num_rollout_workers: int,
            num_cpus_per_worker: int,
            num_gpus_per_worker: int,
            to_send_queue_size: int,
            fragment_length: int,
            resume: Optional[str],
            replay_buffer_config: DictConfig,
            worker_config: DictConfig,
            episode_statistics_config: DictConfig,
    ):
        '''
        This class is responsible for creating and managing a group of rollout workers.

        Args:
            num_rollout_workers (int): number of rollout workers to create
            num_cpus_per_worker (int): number of cpus to allocate for each rollout worker
            fragment_length (int): number of steps in each fragment (this should be equal to the context length of the model)
            replay_buffer_config (DictConfig): configuration for the replay buffer
            worker_config (DictConfig): configuration for rollout workers
            episode_statistics_config (DictConfig): configuration for episode statistics
        '''
        self.num_rollout_workers = num_rollout_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.fragment_length = fragment_length
        self.replay_buffer_config = replay_buffer_config
        self.episode_statistics_config = episode_statistics_config
        self.worker_config = worker_config
        self.policy_generator = policy_generator
        self.env_generator = env_generator
        self.num_gpus_per_worker = num_gpus_per_worker
        self.to_send_queue_size = to_send_queue_size
        self.use_normalized_vf = use_normalized_vf
        
        self.replay_buffer = ReplayBufferInterface(self.replay_buffer_config)

        self.episode_statistics = EpisodeStatistics.remote(
            discount=discount, # type: ignore
            **self.episode_statistics_config
        )

        import pickle
        try:
            sim = env_generator()
            pickle.dumps(sim)
            print("env_generator is pickleable")
            del sim
        except Exception as e:
            ray.util.pdb.set_trace

        self.rollout_workers = [
            RolloutWorkerWrapper.options( # type: ignore
                resources={"rollout_workers": 1}, 
                num_cpus=self.num_cpus_per_worker,
                num_gpus=self.num_gpus_per_worker,
                max_concurrency=WRAPPER_CONCURRENCY
            ).remote(
                fragment_length=self.fragment_length,
                policy_generator=self.policy_generator,
                env_generator=self.env_generator,
                resume = resume,
                worker_config=self.worker_config,
                episode_statistics=self.episode_statistics,
                max_staleness=self.replay_buffer_config.max_staleness,
                use_normalized_vf=self.use_normalized_vf,
                to_send_queue_size=self.to_send_queue_size,
                rollout_worker_id = rollout_worker_id
            )
            for rollout_worker_id in range(self.num_rollout_workers)
        ]
    
    def start(self):
        """
        Starts the rollout and sender threads for all managed RolloutWorkerWrappers.
        """
        for worker in self.rollout_workers:
            worker.rollout_thread.remote()
            worker.sender_thread.remote()

    def update_model(self, session_id: str, model_version: int, packed_state_dict_ref: List[ray.ObjectRef]):
        """
        Updates the model for all managed RolloutWorkerWrappers and the replay buffer.

        :param session_id: The ID of the current training session.
        :param model_version: The version of the model to update to.
        :param packed_state_dict_ref: A list containing a Ray ObjectRef to the packed state dictionary of the model.
        :raises AssertionError: if packed_state_dict_ref does not contain exactly one element.
        """
        assert len(packed_state_dict_ref) == 1
        remotes = [
            worker.update_model.remote(
                session_id=session_id,
                model_version=model_version,
                packed_state_dict_ref=packed_state_dict_ref
            ) for worker in self.rollout_workers
        ]
        ray.get(remotes)
        
        self.replay_buffer.update_model_version(
            session_id=session_id, 
            model_version=model_version
        )

    def log_statistics(self, step: int, record_next_episode: bool):
        """
        Logs statistics through the EpisodeStatistics actor.

        :param step: The current training step.
        :param record_next_episode: A boolean indicating whether to record the next episode.
        """
        print("received_request in menager:"+ str(step)+ str(record_next_episode))
        ray.get(self.episode_statistics.log_statistics.remote(step, record_next_episode))

    def get_replay_buffer_config(self):
        """
        Returns the configuration for the replay buffer.

        :returns: The replay buffer configuration.
        """
        return self.replay_buffer_config
    
    def update_training_session(self):
        """
        Updates the training session for the EpisodeStatistics actor and the replay buffer.
        This typically involves defining metrics for logging.
        """
        ray.get(self.episode_statistics.update_training_session.remote())
        self.replay_buffer.update_training_session()

@ray.remote
class RolloutManager(_RolloutManager):
    """
    A Ray actor that extends _RolloutManager to be remotely accessible.

    This class allows the _RolloutManager to be instantiated as a Ray actor,
    making its methods callable from other Ray actors or the main driver script.
    It also saves its initialization configuration.

    :param kwargs: Keyword arguments passed to the _RolloutManager constructor.
    """
    def __init__(self, **kwargs):
        self.saved_config = kwargs
        super().__init__(**kwargs)

    def get_saved_config(self):
        """
        Returns the saved configuration that was used to initialize the RolloutManager.

        :returns: The saved initialization configuration.
        """
        return self.saved_config
