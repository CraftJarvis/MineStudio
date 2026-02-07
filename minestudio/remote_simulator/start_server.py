import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np
from minestudio.simulator.entry import MinecraftSim

class MinecraftService(rpyc.Service):
    def __init__(self):
        self.sim_instance = None

    def on_connect(self, conn):
        print(f"Client connected: {conn}")

    def on_disconnect(self, conn):
        print(f"Client disconnected: {conn}")
        # 安全机制：如果客户端断开连接（如脚本崩溃），自动关闭游戏进程
        if self.sim_instance:
            print("Cleaning up Minecraft instance...")
            try:
                self.sim_instance.close()
            except Exception as e:
                print(f"Error during cleanup: {e}")
            self.sim_instance = None

    def exposed_create_sim(self, *args, **kwargs):
        """客户端调用此方法来初始化真正的 MinecraftSim"""
        print(f"Initializing MinecraftSim with args={args} kwargs={kwargs}")
        self.sim_instance = MinecraftSim(*args, **kwargs)
        return self.sim_instance

    def exposed_get_sim(self):
        return self.sim_instance

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