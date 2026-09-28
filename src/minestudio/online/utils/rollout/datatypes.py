from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Union
import torch
import logging
from minestudio.online.utils import auto_stack, auto_to_torch
from typing import Optional

logger = logging.getLogger("ray")

@dataclass(frozen=True)
class FragmentIndex:
    """
    Represents a unique identifier for a SampleFragment.

    :param worker_uuid: The unique identifier of the worker that generated the fragment.
    :param fid_in_worker: The fragment's ID within that worker.
    """
    worker_uuid: str
    fid_in_worker: int

@dataclass
class SampleFragment:
    """
    Represents a fragment of a trajectory, containing observations, actions, rewards, and other metadata.

    :param obs: The observation from the environment.
    :param action: The action taken by the agent.
    :param next_done: A boolean indicating whether the episode terminated after this fragment.
    :param reward: The reward received after taking the action.
    :param first: A boolean indicating if this is the first fragment in an episode.
    :param in_state: The recurrent state of the agent.
    :param worker_uuid: The unique identifier of the worker that generated the fragment.
    :param fid_in_worker: The fragment's ID within that worker.
    :param next_obs: The next observation from the environment.
    :param episode_uuids: A list of unique identifiers for the episodes this fragment belongs to.
    """
    obs: Union[Dict[str, Any], torch.Tensor]
    action: Union[Dict[str, Any], torch.Tensor]
    next_done: np.ndarray
    reward: np.ndarray
    first: np.ndarray
    in_state: List[np.ndarray]
    worker_uuid: str
    fid_in_worker: int
    next_obs: Dict[str, Any]
    episode_uuids: List[str]
    @property
    def index(self) -> FragmentIndex:
        """
        Returns the FragmentIndex for this SampleFragment.

        :returns: The FragmentIndex object.
        """
        return FragmentIndex(worker_uuid=self.worker_uuid, fid_in_worker=self.fid_in_worker)
    def print(self) -> None:
        """
        Prints the contents of the SampleFragment for debugging purposes.
        """
        logger.info(f"FragmentIndex: {self.index}, obs: {self.obs}")
        logger.info(f"FragmentIndex: {self.index}, action: {self.action}")
        logger.info(f"FragmentIndex: {self.index}, next_done: {self.next_done}")
        logger.info(f"FragmentIndex: {self.index}, reward: {self.reward}")
        logger.info(f"FragmentIndex: {self.index}, first: {self.first}")
        logger.info(f"FragmentIndex: {self.index}, in_state: {self.in_state}")
        logger.info(f"FragmentIndex: {self.index}, worker_uuid: {self.worker_uuid}")
        logger.info(f"FragmentIndex: {self.index}, fid_in_worker: {self.fid_in_worker}")
        logger.info(f"FragmentIndex: {self.index}, next_obs: {self.next_obs}")
        logger.info(f"FragmentIndex: {self.index}, episode_uuids: {self.episode_uuids}")
    
class FragmentDataDict(Dict[FragmentIndex, Any]):
    """
    A dictionary that maps FragmentIndex to arbitrary data.
    It provides a helper method to format a batch of fragments for model input.
    """
    def format_batch(self, fragments: List[SampleFragment], device: torch.device):
        """
        Formats a list of SampleFragments into a batch suitable for model input.
        It retrieves the corresponding data for each fragment from the dictionary,
        stacks them, and moves them to the specified device.

        :param fragments: A list of SampleFragment objects.
        :param device: The torch device to move the batch to.
        :returns: A batch of data ready for model input.
        """
        return auto_to_torch(
            auto_stack(
                [self[f.index] for f in fragments]
            ),
            device=device
        )
    

@dataclass(frozen=True)
class FragmentMetadata:
    """
    Metadata associated with a SampleFragment.

    :param model_version: The version of the model used to generate this fragment.
    :param session_id: The ID of the training session.
    :param worker_uuid: The unique identifier of the worker that generated the fragment.
    :param fid_in_worker: The fragment's ID within that worker.
    """
    model_version: int
    session_id: str
    worker_uuid: str
    fid_in_worker: int

@dataclass
class StepRecord:
    """
    Represents a single step taken in the environment.

    :param worker_uuid: The unique identifier of the worker that generated this step.
    :param obs: The observation from the environment.
    :param state: The recurrent state of the agent.
    :param action: The action taken by the agent.
    :param last_reward: The reward received from the previous step.
    :param last_terminated: Whether the episode terminated after the previous step.
    :param last_truncated: Whether the episode was truncated after the previous step.
    :param model_version: The version of the model used to generate this step.
    :param episode_uuid: The unique identifier of the episode this step belongs to.
    :param session_id: The ID of the training session.
    """
    worker_uuid: str
    obs: Dict[str, Any]
    state: Optional[List[np.ndarray]]
    action: Dict[str, Any]
    last_reward: float
    last_terminated: bool
    last_truncated: bool
    model_version: int
    episode_uuid: str
    session_id: str