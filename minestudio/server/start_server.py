import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np
from minestudio.simulator.entry import MinecraftSim
import pickle

def sanitize_data(obj):
    """
    更加健壮的递归清理函数，处理非字符串键和无法序列化的对象。
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith('_'):
                continue
            new_dict[k] = sanitize_data(v)
        return new_dict
    
    elif isinstance(obj, (list, tuple)):
        return [sanitize_data(x) for x in obj]
    
    elif isinstance(obj, np.ndarray):
        return obj
    
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    else:
        try:
            return obj
        except Exception:
            return str(obj)

class MinecraftService(rpyc.Service):
    def __init__(self):
        self.sim_instance = None

    def exposed_create_sim(self, *args, **kwargs):
        self.sim_instance = MinecraftSim(*args, **kwargs)
        return "SUCCESS" # 仅返回状态，不传输复杂对象

    def exposed_reset(self):
        if not self.sim_instance:
            raise RuntimeError("Sim not initialized!")
        obs, info = self.sim_instance.reset()
        return sanitize_data(obs), sanitize_data(info)

    def exposed_step(self, action):
        obs, reward, terminated, truncated, info = self.sim_instance.step(action)
        return sanitize_data(obs), reward, terminated, truncated, sanitize_data(info)
    
    def exposed_step_pickled(self, pickled_action):
        action = pickle.loads(pickled_action)
        obs, reward, terminated, truncated, info = self.sim_instance.step(action)
        return sanitize_data(obs), reward, terminated, truncated, sanitize_data(info)

    def exposed_close(self):
        if self.sim_instance:
            self.sim_instance.close()
            self.sim_instance = None

    def exposed_get_spaces(self):
        return self.sim_instance.action_space, self.sim_instance.observation_space

if __name__ == "__main__":
    PORT = 18861
    config = {
        'allow_public_attrs': True, 
        'allow_all_attrs': True, 
        'allow_pickle': True,
        'sync_request_timeout': 300, # 5 分钟超时
    }
    
    print(f"Starting MineStudio RPyC Server on 0.0.0.0:{PORT}...")
    server = ThreadedServer(MinecraftService, port=PORT, protocol_config=config)
    server.start()