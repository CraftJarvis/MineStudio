'''
Date: 2025-05-20 12:09:48
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-05-23 11:39:33
FilePath: /MineStudio/var/minestudio/online/rollout/replay_buffer/fragment_store.py
'''
import ray
import logging
from diskcache import FanoutCache
from uuid import uuid4
from minestudio.online.utils.rollout.datatypes import SampleFragment

class LocalFragmentStoreImpl:
    """
    A local implementation of a fragment store using diskcache.FanoutCache.

    This class provides methods to add, get, delete, and clear fragments
    stored on the local disk.

    :param path: The directory path where the cache will be stored.
    :param num_shards: The number of shards to use for the FanoutCache.
    """
    def __init__(self, 
        path: str,
        num_shards: int,
    ):
        self.cache = FanoutCache(path, shards=num_shards, eviction_policy="none")

    def add_fragment(self, fragment: SampleFragment):
        """
        Adds a fragment to the store and returns a unique ID for it.

        :param fragment: The SampleFragment object to store.
        :returns: A unique string ID for the stored fragment.
        """
        fragment_uuid = str(uuid4())
        self.cache[fragment_uuid] = fragment
        return fragment_uuid
    
    def get_fragment(self, fragment_uuid: str):
        """
        Retrieves a fragment from the store by its unique ID.

        :param fragment_uuid: The unique ID of the fragment to retrieve.
        :returns: The retrieved SampleFragment object.
        """
        return self.cache[fragment_uuid]
    
    def delete_fragment(self, fragment_uuid: str):
        """
        Deletes a fragment from the store by its unique ID.

        :param fragment_uuid: The unique ID of the fragment to delete.
        """
        del self.cache[fragment_uuid]

    def clear(self):
        """
        Removes all fragments from the store.
        """
        self.cache.clear()

    def get_disk_space(self):
        """
        Gets the total disk space used by the cache in bytes.

        :returns: The disk space used by the cache.
        """
        return self.cache.volume()

@ray.remote(resources={"database": 0.0001})
class RemoteFragmentStoreImpl:
    """
    A Ray actor that wraps LocalFragmentStoreImpl to provide a remote fragment store.

    This allows the fragment store to be accessed from different nodes in a Ray cluster.
    It delegates all its methods to an instance of LocalFragmentStoreImpl.

    :param kwargs: Keyword arguments to be passed to the LocalFragmentStoreImpl constructor.
    """
    def __init__(self, **kwargs):
        self.local_impl = LocalFragmentStoreImpl(**kwargs)
    def add_fragment(self, fragment: SampleFragment):
        """
        Adds a fragment to the remote store.

        :param fragment: The SampleFragment object to store.
        :returns: A unique string ID for the stored fragment.
        """
        return self.local_impl.add_fragment(fragment)
    def get_fragment(self, fragment_uuid: str):
        """
        Retrieves a fragment from the remote store by its unique ID.

        :param fragment_uuid: The unique ID of the fragment to retrieve.
        :returns: The retrieved SampleFragment object.
        """
        return self.local_impl.get_fragment(fragment_uuid)
    def delete_fragment(self, fragment_uuid: str):
        """
        Deletes a fragment from the remote store by its unique ID.

        :param fragment_uuid: The unique ID of the fragment to delete.
        """
        return self.local_impl.delete_fragment(fragment_uuid)
    def clear(self):
        """
        Removes all fragments from the remote store.
        """
        return self.local_impl.clear()
    def get_disk_space(self):
        """
        Gets the total disk space used by the remote cache in bytes.

        :returns: The disk space used by the cache.
        """
        return self.local_impl.get_disk_space()
    
class FragmentStore:
    """
    A class that provides an interface to either a local or a remote fragment store.

    It checks if the current Ray node has a "database" resource. If so, it uses
    a LocalFragmentStoreImpl. Otherwise, it uses a RemoteFragmentStoreImpl actor.

    :param kwargs: Keyword arguments to be passed to the underlying store implementation (LocalFragmentStoreImpl or RemoteFragmentStoreImpl).
    :raises AssertionError: if the local status cannot be determined.
    """
    def __init__(self, **kwargs):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local = None
        for node in ray.nodes():
            if node["NodeID"] == self.node_id:
                resources = node["Resources"]
                if resources.get("database", 0) > 0:
                    self.local = True
                else:
                    logging.warn("Remote fragment store has not been tested yet")
                    self.local = False
                break

        assert self.local is not None
                
        if not self.local:
            self.remote_impl = RemoteFragmentStoreImpl.options(
                placement_group=None,
                resources={"database": 0.0001}
            ).remote(**kwargs) # type: ignore
        else:
            self.local_impl = LocalFragmentStoreImpl(**kwargs)
    
    def add_fragment(self, fragment: SampleFragment):
        """
        Adds a fragment to the store (either local or remote).

        :param fragment: The SampleFragment object to store.
        :returns: A unique string ID for the stored fragment.
        """
        if self.local:
            return self.local_impl.add_fragment(fragment)
        else:
            return ray.get(self.remote_impl.add_fragment.remote(fragment)) # type: ignore
        
    def get_fragment(self, fragment_uuid: str) -> SampleFragment:
        """
        Retrieves a fragment from the store (either local or remote) by its unique ID.

        :param fragment_uuid: The unique ID of the fragment to retrieve.
        :returns: The retrieved SampleFragment object.
        """
        if self.local:
            return self.local_impl.get_fragment(fragment_uuid) # type: ignore
        else:
            return ray.get(self.remote_impl.get_fragment.remote(fragment_uuid)) # type: ignore
        
    def delete_fragment(self, fragment_uuid: str):
        """
        Deletes a fragment from the store (either local or remote) by its unique ID.

        :param fragment_uuid: The unique ID of the fragment to delete.
        """
        if self.local:
            return self.local_impl.delete_fragment(fragment_uuid)
        else:
            return ray.get(self.remote_impl.delete_fragment.remote(fragment_uuid)) # type: ignore
    
    def clear(self):
        """
        Removes all fragments from the store (either local or remote).
        """
        if self.local:
            return self.local_impl.clear()
        else:
            return ray.get(self.remote_impl.clear.remote()) # type: ignore

    def get_disk_space(self):
        """
        Gets the total disk space used by the cache (either local or remote) in bytes.

        :returns: The disk space used by the cache.
        """
        if self.local:
            return self.local_impl.get_disk_space()
        else:
            return ray.get(self.remote_impl.get_disk_space.remote()) # type: ignore