Here is an example that shows how to load and use OpenAI's VPT policy within the Minecraft environment provided by MineStudio.

```python
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import load_vpt_policy, VPTPolicy

# Option 1: Load the policy from local model files
# Ensure you have the .model (architecture) and .weights (parameters) files.
policy = load_vpt_policy(
    model_path="/path/to/foundation-model-2x.model", 
    weights_path="/path/to/foundation-model-2x.weights"
).to("cuda") # Move the policy to GPU if available

# Option 2: Load the policy from the Hugging Face Model Hub
# This is a convenient way to get pre-trained models.
# policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x").to("cuda")

# Set the policy to evaluation mode. This is important for consistent behavior during inference.
policy.eval()

# Initialize the Minecraft Simulator
# obs_size specifies the resolution of the visual observations.
# callbacks allow for custom actions during the simulation, e.g., recording.
env = MinecraftSim(
    obs_size=(128, 128), 
    callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")]
)

# `memory` stores the recurrent state of the policy (e.g., for RNNs).
# For policies without memory (Markovian), it can be initialized to None.
memory = None 
obs, info = env.reset() # Reset the environment to get the initial observation.

# Simulation loop
for i in range(1200): # Run for 1200 steps
    # Get an action from the policy.
    # `obs`: The current observation from the environment.
    # `memory`: The current recurrent state of the policy.
    # `input_shape='*'`: Indicates that `obs` is a single sample (not a batch or time sequence).
    # The policy handles internal batching/unbatching for its forward pass.
    action, memory = policy.get_action(obs, memory, input_shape='*')
    
    # Apply the action to the environment.
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if the episode has ended.
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(i+1))
        break

env.close() # Close the environment when done.
```

```{hint}
In this example, if `RecordCallback` is used, the recorded video will be saved in the `./output` directory. The `memory` variable handles the policy's recurrent state, and `input_shape='*'` in `get_action` is typical for single-instance inference.
```
