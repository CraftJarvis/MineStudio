import ray
from omegaconf import DictConfig
from minestudio.online.rollout.replay_buffer.fragment_store import FragmentStore
from minestudio.online.rollout.replay_buffer.actor import ReplayBufferActor
from minestudio.online.utils.rollout.datatypes import FragmentMetadata
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment
from typing import List, Optional, Tuple

class ReplayBufferInterface:
    """
    Provides an interface to interact with the ReplayBufferActor.

    This class handles the creation or connection to a ReplayBufferActor named "replay_buffer".
    It also initializes a FragmentStore based on the actor's database configuration.
    All methods to interact with the replay buffer (add, load, fetch fragments, update model version)
    are routed through the ReplayBufferActor.

    :param config: Optional DictConfig. If provided, a new ReplayBufferActor is created with this config.
                   If None, it attempts to connect to an existing actor named "replay_buffer".
    :raises ValueError: If config is provided but an actor already exists, or if config is None and no actor exists.
    """
    def __init__(self, config: Optional[DictConfig] = None):
        existing_actor = None
        try:
            existing_actor = ray.get_actor("replay_buffer")
        except ValueError:
            pass

        if config is not None:
            if existing_actor is not None:
                raise ValueError("Replay buffer already initialized")
            self.actor = ReplayBufferActor.options(name="replay_buffer").remote(** config) # type: ignore
        else:
            if existing_actor is None:
                raise ValueError("Replay buffer not initialized")
            self.actor = existing_actor

        self.database_config = ray.get(self.actor.get_database_config.remote())
        self.store = FragmentStore(** self.database_config)

    def update_training_session(self):
        """
        Calls the update_training_session method of the ReplayBufferActor.

        :returns: The result of the actor's method call.
        """
        return ray.get(self.actor.update_training_session.remote())

    def add_fragment(self, fragment: SampleFragment, metadata: FragmentMetadata):
        """
        Adds a fragment to the FragmentStore and then informs the ReplayBufferActor.

        :param fragment: The SampleFragment to add.
        :param metadata: The FragmentMetadata associated with the fragment.
        """
        fragment_id = self.store.add_fragment(fragment)

        ray.get(
            self.actor.add_fragment.remote(
                fragment_id=fragment_id,
                metadata=metadata,
            )
        )

    def load_fragment(self, fragment_id: str) -> SampleFragment:
        """
        Loads a fragment directly from the FragmentStore.

        :param fragment_id: The unique ID of the fragment to load.
        :returns: The loaded SampleFragment.
        """
        return self.store.get_fragment(fragment_id)
    
    def fetch_fragments(self, num_fragments: int) -> List[Tuple[FragmentIndex, str]]:
        """
        Fetches a list of fragment IDs and their indices from the ReplayBufferActor.

        :param num_fragments: The number of fragments to fetch.
        :returns: A list of tuples, each containing a FragmentIndex and the fragment_id.
        """
        return ray.get(
            self.actor.fetch_fragments.remote(num_fragments=num_fragments)
        )
    
    def prepared_fragments(self) -> List[Tuple[FragmentIndex, str]]:
        """
        Retrieves the fragments that were prepared by the last call to fetch_fragments in the ReplayBufferActor.

        :returns: A list of tuples, each containing a FragmentIndex and the fragment_id.
        """
        return ray.get(
            self.actor.prepared_fragments.remote()
        )
    
    def update_model_version(self, session_id: str, model_version: int):
        """
        Updates the model version in the ReplayBufferActor.

        :param session_id: The ID of the current training session.
        :param model_version: The new model version.
        :returns: The result of the actor's method call.
        """
        return ray.get(
            self.actor.update_model_version.remote(
                session_id=session_id,
                model_version=model_version
            )
        )