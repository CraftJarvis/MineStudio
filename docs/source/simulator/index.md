# Simulator

We provide an easily customizable Minecraft simulator that is developed based on [MineRL](https://github.com/minerllabs/minerl). We designed a Gym-style Minecraft Wrapper, which supports a callbacks mechanism, allowing users to customize their own environment, including custom reward functions, environment initialization, trajectory recording, and more. 

```{toctree}
:caption: MineStudio Simulator

general-information
design-principles
play
```

## Quick Start

```{include} quick-simulator.md
```

## Basic Arguments

The simulator has several arguments that can be used to customize the environment. 

| Argument | Default | Description |
| --- | --- | --- |
| action_type | "agent" | The style of the action space. Can be 'env' or 'agent'. |
| obs_size | (224, 224) | The resolution of the observation (cv2 resize). |
| render_size | (640, 360) | The original resolution of the game. |
| seed | 0 | The seed of the Minecraft world. |
| inventory | {} | The initial inventory of the agent. |
| preferred_spawn_biome | None | The preferred spawn biome when calling reset. |
| num_empty_frames | 20 | The number of empty frames to skip when calling reset. |
| callbacks | [] | A list of callbacks to customize the environment (**advanced**). |
| camera_config | `CameraConfig()` | Configuration for camera quantization and binning settings. See `CameraConfig` class for details. |

## Using Callbacks

Callbacks can be used to customize the environment in a flexible way. We provide several built-in callbacks, and users can also implement their own callbacks. 

Here is some examples of how to use callbacks:

```{note}
All the callbacks are optional, and you can use them in any combination. 
```

### Test Speed
We provide a callback to test the speed of the simulator by hooking into the `before_step` and `after_step` methods. 
```python
from minestudio.simulator.callbacks import SpeedTestCallback

sim = MinecraftSim(callbacks=[SpeedTestCallback(interval=50)])
```

If you run the above code, you will see the following output:
```
Speed Test Status: 
Average Time: 0.03 
Average FPS: 38.46 
Total Steps: 50 

Speed Test Status: 
Average Time: 0.02 
Average FPS: 45.08 
Total Steps: 100 
```

### Send Chat Commands

We provide a callback to send chat commands to the Minecraft server **when the environment is reset**. This enable users to customize the environment initialization. 
```python
from minestudio.simulator.callbacks import CommandsCallback

sim = MinecraftSim(callbacks=[CommandsCallback(commands=[
    "/time set day", 
    "/weather clear",
    "/give @p minecraft:iron_sword 1",
    "/give @p minecraft:diamond 64",
])])
```

```{hint}
The commands will be executed in the order they are provided. 
```

### Custom Reward Function

We provide a simple callback to customize the reward function. This callback will be called after each step, and detect the event in ``info`` to calculate the reward. 
```python
from minestudio.simulator.callbacks import RewardsCallback

sim = MinecraftSim(callbacks=[
    RewardsCallback([{
        'event': 'kill_entity', 
        'objects': ['cow', 'sheep'], 
        'reward': 1.0, 
        'identity': 'kill sheep or cow', 
        'max_reward_times': 5, 
    }])
])
```
```{hint}
This example will give a reward of 1.0 when the agent kills a sheep or cow. The maximum reward times is 5. 
```

### Fast Reset

Generally, the environment reset is slow because it needs to restart the Minecraft server. We provide a callback to speed up the reset process by hooking into the `before_reset` method. This would be pretty useful when you want to reset the environment frequently, like in RL training.  
```python
from minestudio.simulator.callbacks import FastResetCallback

sim = MinecraftSim(callbacks=[
    FastResetCallback(
        biomes=['mountains'],
        random_tp_range=1000,
    ), 
])
```
```{hint}
Fast reset is implemented by killing the agent and teleporting it to a random location. 
```

### Record Trajectories

We provide a callback to record trajectories. It will records the observation, action, info at each step, and save them to a specified path. 
```python
from minestudio.simulator.callbacks import RecordCallback

sim = MinecraftSim(callbacks=[RecordCallback(record_path="./output", fps=30)])
```
```{hint}
You can use this callback to record the agent's behavior and analyze it later. Or you can use it to generate a dataset for imitation learning. 
```

```{button-ref}  ./design-principles
:color: primary
:outline:
:expand:
Learn more about MineStudio Simulator callbacks
```