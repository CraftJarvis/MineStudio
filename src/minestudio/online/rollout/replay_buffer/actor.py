import asyncio
from collections import defaultdict, deque
from omegaconf import DictConfig
import ray
from typing import Deque, List, Dict, Any, Tuple, cast
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment
from minestudio.online.rollout.replay_buffer.fragment_store import FragmentStore
from minestudio.online.utils.rollout.datatypes import FragmentMetadata
import minestudio.online.utils.train.wandb_logger as wandb_logger
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import logging
import torchmetrics

logger = logging.getLogger("ray")

class FragmentRecord:
    """
    Represents a record of a fragment with its metadata and a reference to the FragmentManager.

    This class is used to track references to fragments in the FragmentStore.
    When a FragmentRecord instance is deleted (no longer referenced),
    it informs the FragmentManager to potentially clean up the fragment from the store
    if its reference count drops to zero.

    :param fragment_id: The unique identifier of the fragment.
    :param metadata: The FragmentMetadata associated with this fragment.
    :param manager: The FragmentManager instance that manages this fragment.
    """
    def __init__(self,
        fragment_id: str,
        metadata: FragmentMetadata,
        manager,
    ):
        self.fragment_id = fragment_id
        self.metadata = metadata
        self.manager = manager
        self.manager.ref_count[fragment_id] += 1

    def __del__(self):
        """
        Decrements the reference count of the fragment in the FragmentManager.
        If the reference count becomes zero, it triggers the cleaning process in the manager.
        """
        self.manager.ref_count[self.fragment_id] -= 1
        if self.manager.ref_count[self.fragment_id] == 0:
            self.manager.clean(self.fragment_id)

class FragmentManager:
    """
    Manages fragments stored in a FragmentStore, primarily by tracking their reference counts.

    :param fragment_store: An instance of FragmentStore where fragments are physically stored.
    """
    def __init__(self, fragment_store: FragmentStore):
        self.fragment_store = fragment_store
        self.ref_count = defaultdict(int)

    def create_fragment_record(self, fragment_id: str, metadata: FragmentMetadata):
        """
        Creates a new FragmentRecord for a given fragment_id and its metadata.

        :param fragment_id: The unique identifier of the fragment.
        :param metadata: The FragmentMetadata associated with this fragment.
        :returns: A new FragmentRecord instance.
        """
        return FragmentRecord(
            fragment_id=fragment_id,
            metadata=metadata,
            manager=self
        )

    def clean(self, fragment_id: str):
        """
        Removes a fragment from the reference count and deletes it from the FragmentStore.

        This method is called when a fragment's reference count drops to zero.

        :param fragment_id: The unique identifier of the fragment to clean.
        """
        del self.ref_count[fragment_id]
        self.fragment_store.delete_fragment(fragment_id)

@dataclass
class ChunkRecord:
    """
    Represents a chunk of fragment records, along with model version, session ID, and use count.

    :param fragment_records: A list of FragmentRecord objects that form this chunk.
    :param model_version: The model version associated with the fragments in this chunk.
    :param session_id: The session ID associated with the fragments in this chunk.
    :param use_count: How many times this chunk has been used for training.
    """
    fragment_records: List[FragmentRecord]
    model_version: int
    session_id: str
    use_count: int

@ray.remote
class ReplayBufferActor:
    """
    A Ray actor that implements a replay buffer for storing and sampling experience fragments.

    This actor manages chunks of fragments, handles staleness and reuse of data,
    and provides an interface for adding new fragments and fetching batches for training.

    :param max_chunks: The maximum number of chunks to store in the replay buffer.
    :param max_staleness: The maximum allowed difference between the current model version and a fragment's model version for it to be considered valid.
    :param max_reuse: The maximum number of times a chunk can be reused for training before being discarded.
    :param database_config: A DictConfig object for configuring the FragmentStore (database).
    :param fragments_per_chunk: The number of fragments that constitute a single chunk.
    :param fragments_per_report: Optional. If set, logs statistics every N fragments added.
    """
    def __init__(self,
        max_chunks: int,
        max_staleness: int,
        max_reuse: int,
        database_config: DictConfig,
        fragments_per_chunk: int,
        fragments_per_report: Optional[int] = None,
    ):
        self.fragments_per_report = fragments_per_report

        self.fragment_added = 0
        self._last_report_time = time.time()

        self.max_reuse = max_reuse
        self.current_model_version = -1
        self.current_session_id = ""

        self.max_staleness = max_staleness
        self.fragments_per_chunk = fragments_per_chunk

        self.database_config = database_config
        self.fragment_store = FragmentStore(** database_config) # type: ignore
        self.fragment_store.clear()
        
        self.fragment_manager = FragmentManager(self.fragment_store)

        self.max_chunks = max_chunks
        self.chunks: Deque[ChunkRecord] = deque()

        self.recv_buffer: Dict[str, Deque[FragmentRecord]] = {}
        self.fragments_to_return: List[FragmentRecord] = []

        self.fetch_reqs: List[Tuple[int, asyncio.Event]] = []

        self.reuse_metric = torchmetrics.MeanMetric()
        self.staleness_metric = torchmetrics.MeanMetric()

    def update_training_session(self):
        """
        Placeholder for any logic needed when a new training session starts.
        Currently does nothing.
        """
        pass

    def pop_chunk(self) -> None:
        """
        Removes the oldest chunk from the buffer and updates reuse metrics.
        """
        self.reuse_metric.update(self.chunks[0].use_count, self.fragments_per_chunk)
        self.chunks.popleft()

    def add_chunk(self, chunk_record: ChunkRecord) -> None:
        """
        Adds a new chunk to the replay buffer.

        If the buffer is full, the oldest chunk is popped.
        Chunks that are too stale or belong to a different session are discarded immediately.
        Notifies any pending fetch requests if enough chunks become available.

        :param chunk_record: The ChunkRecord to add.
        """
        while len(self.chunks) >= self.max_chunks:
            self.pop_chunk()

        if (self.current_model_version - chunk_record.model_version <= self.max_staleness
            and self.current_session_id == chunk_record.session_id):
            self.chunks.append(chunk_record)

            new_fetch_reqs = []
            for req, evt in self.fetch_reqs:
                if len(self.chunks) >= req:
                    evt.set()
                else:
                    new_fetch_reqs.append((req, evt))
            self.fetch_reqs = new_fetch_reqs
        else:
            self.reuse_metric.update(chunk_record.use_count, self.fragments_per_chunk)

    async def add_fragment(self, fragment_id: str, metadata: FragmentMetadata):
        """
        Adds a single fragment to the replay buffer.

        Fragments are buffered per worker_uuid. Once enough fragments are collected
        from a worker to form a chunk (fragments_per_chunk), a ChunkRecord is created
        and added to the main buffer via `add_chunk`.
        Handles stale fragments by discarding them from the worker's receive buffer.
        Logs statistics periodically if `fragments_per_report` is set.

        :param fragment_id: The unique ID of the fragment to add.
        :param metadata: The FragmentMetadata associated with the fragment.
        """
        fragment_record = self.fragment_manager.create_fragment_record(
            fragment_id=fragment_id,
            metadata=metadata
        )

        worker_uuid = metadata.worker_uuid
        if worker_uuid in self.recv_buffer:
            assert self.recv_buffer[worker_uuid][-1].metadata.fid_in_worker + 1 == metadata.fid_in_worker
            self.recv_buffer[worker_uuid].append(fragment_record)
        else:
            self.recv_buffer[worker_uuid] = deque([fragment_record])

        while len(self.recv_buffer[worker_uuid]) > 0 and (
            self.current_model_version - self.recv_buffer[worker_uuid][0].metadata.model_version > self.max_staleness
            or
            self.current_session_id != self.recv_buffer[worker_uuid][0].metadata.session_id
        ):
            self.reuse_metric.update(0.0, self.fragments_per_chunk)

            self.recv_buffer[worker_uuid].popleft()
        
        if len(self.recv_buffer[worker_uuid]) == 0:
            self.recv_buffer.pop(worker_uuid)
        elif len(self.recv_buffer[worker_uuid]) >= self.fragments_per_chunk:
            fragment_records = list(self.recv_buffer[worker_uuid])
            chunk_record = ChunkRecord(
                fragment_records=fragment_records,
                model_version=fragment_records[0].metadata.model_version,
                session_id=fragment_records[0].metadata.session_id,
                use_count=0,
            )
            self.recv_buffer.pop(worker_uuid)
            self.add_chunk(chunk_record)

        self.fragment_added += 1
        now = time.time()
        if self.fragments_per_report and self.fragment_added % self.fragments_per_report == 0:
            info = {
                "replay_buffer/fragments_per_second": self.fragments_per_report / (now - self._last_report_time),
                "replay_buffer/replay_buffer_size (chunk)": len(self.chunks),
                "replay_buffer/model_version": self.current_model_version
            }
            wandb_logger.log(info)
            self._last_report_time = now

    async def update_model_version(self, session_id: str, model_version: int) -> None:
        """
        Updates the current model version and session ID for the replay buffer.

        This triggers the removal of any chunks that have become too stale
        or belong to a previous session.

        :param session_id: The new session ID.
        :param model_version: The new model version.
        """
        self.current_model_version = model_version
        self.current_session_id = session_id
        while (len(self.chunks) > 0 and self.chunks[0].model_version < self.current_model_version - self.max_staleness
               or len(self.chunks) > 0 and self.chunks[0].session_id != self.current_session_id):
            self.pop_chunk()

    async def fetch_fragments(self, num_fragments: int) -> List[Tuple[FragmentIndex, str]]:
        """
        Fetches a specified number of fragments for training.

        Randomly samples chunks from the buffer. If not enough chunks are available,
        it waits until they are. Tracks the reuse count of chunks and discards them
        if `max_reuse` is reached. Logs reuse and staleness metrics.

        :param num_fragments: The total number of fragments to fetch.
                             Must be a multiple of `fragments_per_chunk`.
        :returns: A list of tuples, each containing a FragmentIndex and the fragment_id.
        :raises AssertionError: if num_fragments is not a multiple of fragments_per_chunk
                               or if num_chunks requested exceeds max_chunks.
        """
        assert num_fragments % self.fragments_per_chunk == 0
        num_chunks = num_fragments // self.fragments_per_chunk
        assert num_chunks <= self.max_chunks

        while len(self.chunks) < num_chunks:
            evt = asyncio.Event()
            self.fetch_reqs.append((num_chunks, evt))
            await evt.wait()

        selected_idxs = np.random.choice(
            list(range(len(self.chunks))),
            num_chunks,
            replace=False
        )

        chunks_list: List[Optional[ChunkRecord]] = list(self.chunks)
        self.fragments_to_return = []
        for idx in selected_idxs:
            assert chunks_list[idx] is not None
            self.fragments_to_return += chunks_list[idx].fragment_records
            chunks_list[idx].use_count += 1
            if chunks_list[idx].use_count >= self.max_reuse:
                self.reuse_metric.update(chunks_list[idx].use_count, self.fragments_per_chunk)
                self.staleness_metric.update(self.current_model_version - chunks_list[idx].model_version, self.fragments_per_chunk)
                chunks_list[idx] = None
        self.chunks = deque([chunk for chunk in chunks_list if chunk is not None])

        wandb_logger.log({
            "replay_buffer/avg_reuse": self.reuse_metric.compute(),
            "replay_buffer/avg_staleness": self.staleness_metric.compute()
        })
        self.reuse_metric.reset()
        self.staleness_metric.reset()
                
        return [(
                    FragmentIndex(worker_uuid=fragment_record.metadata.worker_uuid, fid_in_worker=fragment_record.metadata.fid_in_worker),
                    fragment_record.fragment_id
                ) for fragment_record in self.fragments_to_return]
    
    async def get_database_config(self):
        """
        Returns the database configuration used by the FragmentStore.

        :returns: The database configuration (DictConfig).
        """
        return self.database_config

    async def prepared_fragments(self) -> List[Tuple[FragmentIndex, str]]:
        """
        Returns the list of fragments that were prepared by the last call to `fetch_fragments`.

        :returns: A list of tuples, each containing a FragmentIndex and the fragment_id.
        """
        return [(
                    FragmentIndex(worker_uuid=fragment_record.metadata.worker_uuid, fid_in_worker=fragment_record.metadata.fid_in_worker),
                    fragment_record.fragment_id
                ) for fragment_record in self.fragments_to_return]