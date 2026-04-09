# 文件名: minestudio/simulator/entry_server.py
# 部署位置: BOYA

import gymnasium
import rpyc
import numpy as np
from typing import Dict, Any, Tuple, Literal, List, Optional
from minestudio.simulator.entry import CameraConfig, MinecraftCallback
import pickle

def _sanitize_action_for_rpc(action):
    """
    将 action 转换为可以安全通过 RPyC 传输的格式。
    剥离所有的 Torch Tensor 或复杂的引用。
    """
    import torch
    import numpy as np

    if isinstance(action, dict):
        return {k: _sanitize_action_for_rpc(v) for k, v in action.items()}
    elif isinstance(action, (list, tuple)):
        return [_sanitize_action_for_rpc(x) for x in action]
    elif hasattr(action, 'detach'):  # 捕获 Torch Tensor
        # 如果是 tensor，转成 numpy
        return action.detach().cpu().numpy()
    elif isinstance(action, np.ndarray):
        # 确保是连续内存的 numpy 数组，有时能避免 rpyc 序列化问题
        return np.ascontiguousarray(action)
    else:
        return action

class MinecraftSimRemote(gymnasium.Env):
    def __init__(self, host, port, **kwargs):
        super().__init__()
        # 1. 建立连接
        self.conn = rpyc.connect(host, int(port), config={
                'allow_pickle': True, 
                'sync_request_timeout': 300,
                'allow_public_attrs': True,
                'allow_all_attrs': True
            }
        )
        
        # 2. 通知 Server 创建实例
        self.conn.root.create_sim(**kwargs)
        
        # 3. 缓存 space 信息（避免每次都跨网络读取）
        aspace, ospace = self.conn.root.get_spaces()
        self._action_space = rpyc.utils.classic.obtain(aspace)
        self._observation_space = rpyc.utils.classic.obtain(ospace)
        print("Remote session established.")

    def reset(self, **kwargs):
        # 像 step 一样，直接调用服务端的 reset
        res = self.conn.root.reset()
        # 关键：必须 obtain 转换回本地 numpy 格式，否则 policy 无法计算
        obs, info = rpyc.utils.classic.obtain(res)
        return obs, info

    def step(self, action):
        pickled_action = pickle.dumps(action)
        res = self.conn.root.step_pickled(pickled_action)
        obs, reward, terminated, truncated, info = rpyc.utils.classic.obtain(res)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.conn.root.close()
        self.conn.close()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space