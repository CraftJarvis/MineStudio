import subprocess
from dataclasses import dataclass
import os
import shutil
import diskcache
from pathlib import Path
import signal
import psutil
import sys
import time
from rich.console import Console

# DATABASE_DIR = str(Path(__file__).parent.parent / "tmp" / "database")
import socket
from minestudio.utils.temp import get_mine_studio_dir
hostname = socket.gethostname()
DATABASE_DIR = str(Path(get_mine_studio_dir()) / f"{hostname}_tmp" / "database")


RESETTING_EXPIRE = 5 * 60
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR, exist_ok=True)

if not os.environ.get("MINESTUDIO_DISABLE_DISKCACHE", False):
    # Console().log("Set JARVISBASE_DISABLE_DISKCACHE = False")
    database = diskcache.Cache(DATABASE_DIR, eviction_policy='none')
    instance_list_lock = diskcache.Lock(database, ("lock", "instance_list"), expire=60)
    counter_lock = diskcache.Lock(database, ("lock", "counter"), expire=60)
    daemon_lock = diskcache.Lock(database, ("lock", "daemon"), expire=60)
    resetting_lock = diskcache.Lock(database, ("lock", "resetting"), expire=60)

    if not "instance_list" in database:
        database["instance_list"] = []
    if not "counter" in database:
        database["counter"] = 0
    if not "resetting_list" in database:
        database["resetting_list"] = []

@dataclass
class InstanceRecord:
    minecraft_pgid: int
    parent_pid: int
    uuid: str
    tmp_path: str

def check_daemon():
    with daemon_lock:
        if "daemon" in database:
            daemon_pid, parent_pid = database["daemon"]
            if psutil.pid_exists(daemon_pid) and psutil.pid_exists(parent_pid):
                return
        daemon_pid = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "daemon.py"), str(os.getpid())], 
            close_fds=True,
            start_new_session=True
        ).pid
        database["daemon"] = (daemon_pid, os.getpid())

def counter():
    with counter_lock:
        ret = database["counter"]
        database["counter"] = ret + 1
    return ret

def get_instance_num():
    return len(database["instance_list"])

def check_instance(uuid: str, force_remove: bool = False):
    lock = diskcache.Lock(database, ("lock", uuid), expire=60)
    with lock:
        if not uuid in database:
            return
        record: InstanceRecord = database[uuid]
        if force_remove or record.parent_pid < 0 or not psutil.pid_exists(record.parent_pid):
            record.parent_pid = -1
            database[uuid] = record
            try:
                if record.minecraft_pgid >= 0:
                    try:
                        os.killpg(record.minecraft_pgid, signal.SIGKILL)
                    except:
                        pass
                if os.path.exists(record.tmp_path):
                    shutil.rmtree(record.tmp_path)
                assert not os.path.exists(record.tmp_path)
                with instance_list_lock:
                    instance_list: list = database["instance_list"]
                    instance_list.remove(record.uuid)
                    database["instance_list"] = instance_list
                del database[record.uuid]
            except Exception as e:
                pass

def collect_garbage():
    instance_list: list = database["instance_list"]
    for instance_uuid in instance_list:
        check_instance(instance_uuid)

def write_instance_record(record: InstanceRecord):
    with instance_list_lock:
        instance_list: list = database["instance_list"]
        if not record.uuid in instance_list:
            instance_list.append(record.uuid)
        database["instance_list"] = instance_list
    database[record.uuid] = record

def update_resetting_envs():
    with resetting_lock:
        _resetting_list = database["resetting_list"]
        resetting_list = []
        for request_uuid in _resetting_list:
            if (request_uuid, "resetting_request_time") in database:
                if time.time() - database[(request_uuid, "resetting_request_time")] < RESETTING_EXPIRE:
                    resetting_list.append(request_uuid)
                else:
                    del database[(request_uuid, "resetting_request_time")]
        database["resetting_list"] = resetting_list

def add_resetting_env(request_uuid: str, limit: int = -1):
    update_resetting_envs()
    with resetting_lock:
        if len(database["resetting_list"]) >= limit and limit >= 0:
            return False
        database[(request_uuid, "resetting_request_time")] = time.time()
        resetting_list = database["resetting_list"]
        if not request_uuid in resetting_list:
            resetting_list.append(request_uuid)
            database["resetting_list"] = resetting_list
        return True

def remove_resetting_env(request_uuid: str):
    with resetting_lock:
        if (request_uuid, "resetting_request_time") in database:
            del database[(request_uuid, "resetting_request_time")]
        if request_uuid in database["resetting_list"]:
            resetting_list = database["resetting_list"]
            resetting_list.remove(request_uuid)
            database["resetting_list"] = resetting_list