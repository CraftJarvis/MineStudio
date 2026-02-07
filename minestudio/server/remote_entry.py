# 文件名: minestudio/simulator/entry_server.py
# 部署位置: BOYA

import gymnasium
import rpyc
import numpy as np
from typing import Dict, Any, Tuple, Literal, List, Optional
from minestudio.simulator.entry import CameraConfig, MinecraftCallback

SERVER_HOST = "172.17.40.12"
SERVER_PORT = 18861

class MinecraftSimRemote(gymnasium.Env):
    """
    MinecraftSim 的远程代理版本。
    API 与 MinecraftSim 保持一致。
    """
    def __init__(
        self,   
        host: str = SERVER_HOST,
        port: int = SERVER_PORT,
        # 以下参数与 MinecraftSim 保持一致
        action_type: Literal['env', 'agent'] = 'agent',
        obs_size: Tuple[int, int] = (224, 224),
        render_size: Tuple[int, int] = (640, 360),
        seed: int = 0,
        inventory: Dict = {},
        preferred_spawn_biome: Optional[str] = None,
        num_empty_frames: int = 20,
        callbacks: List[MinecraftCallback] = [],
        camera_config: CameraConfig = None,
        **kwargs
    ) -> Any:
        super().__init__()
        
        # 1. 建立连接
        print(f"Connecting to remote simulator at {host}:{port}...")
        self.conn = rpyc.connect(host, port, config={
            'allow_pickle': True, 
            'sync_request_timeout': 300,
            'allow_public_attrs': True,
            'allow_all_attrs': True
        })
        
        # 2. 在远程服务器上创建实例
        self.remote_sim = self.conn.root.create_sim(
            action_type=action_type,
            obs_size=obs_size,
            render_size=render_size,
            seed=seed,
            inventory=inventory,
            preferred_spawn_biome=preferred_spawn_biome,
            num_empty_frames=num_empty_frames,
            callbacks=callbacks,
            camera_config=camera_config,
            **kwargs
        )
        print("Remote Minecraft instance initialized.")

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 调用远程 step
        obs, reward, terminated, truncated, info = self.remote_sim.step(action)
        
        # [关键优化] 使用 rpyc.utils.classic.obtain 将远程对象(NetRef)转换为本地对象
        # 如果不加这一步，访问 obs['image'] 会极其缓慢，因为每次像素读取都会触发网络请求
        obs = rpyc.utils.classic.obtain(obs)
        info = rpyc.utils.classic.obtain(info)
        
        return obs, reward, terminated, truncated, info

    def reset(self) -> Tuple[np.ndarray, Dict]:
        obs, info = self.remote_sim.reset()
        obs = rpyc.utils.classic.obtain(obs)
        info = rpyc.utils.classic.obtain(info)
        
        return obs, info

    def render(self) -> None:
        image = self.remote_sim.render()
        return rpyc.utils.classic.obtain(image)

    def close(self) -> None:
        if hasattr(self, 'remote_sim'):
            self.remote_sim.close()
        if hasattr(self, 'conn'):
            self.conn.close()

    def noop_action(self) -> Dict[str, Any]:
        return rpyc.utils.classic.obtain(self.remote_sim.noop_action())

    @property
    def action_space(self):
        # 缓存 space 对象，避免重复网络请求
        if not hasattr(self, '_action_space'):
            self._action_space = rpyc.utils.classic.obtain(self.remote_sim.action_space)
        return self._action_space

    @property
    def observation_space(self):
        if not hasattr(self, '_observation_space'):
            self._observation_space = rpyc.utils.classic.obtain(self.remote_sim.observation_space)
        return self._observation_space
    
    def __getattr__(self, name):
        """代理其他所有未显式定义的方法到远程对象"""
        return getattr(self.remote_sim, name)