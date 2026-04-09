from minestudio.simulator.entry import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback
from minestudio.models import VPTPolicy, load_vpt_policy

if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
    ).to("cuda")
    
    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="forest", 
        callbacks=[
            RecordCallback(record_path="./output", fps=30, frame_type="pov"),
            SpeedTestCallback(50),
        ],
        host = "172.17.40.11",
        port = "18861"
    )
    memory = None
    obs, info = env.reset()


    for i in range(600):
        action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
    env.reset()
    print("Resetting the environment")
    for i in range(600):
        action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()