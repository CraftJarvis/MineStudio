# Graphical User Interfaces

## Hello World

Here is a minimal example of launching a playable Minecraft GUI. The code can only be run with a display, so make sure you have a display available.
```python
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import PlayCallback

sim = MinecraftSim(
    action_type="env",
    callbacks=[
        PlayCallback()
    ]
)

obs, info = sim.reset()
terminated = False

while not terminated:
    action = None
    obs, reward, terminated, info = sim.step(action)

sim.close()
```
A window will pop up like this, and you can start playing Minecraft in the GUI. When `action` in `sim.step(action)` is `None`, the player will be controlled by the keyboard or a callback function.
<img src="../_static/image/gui.png" alt="Alt text" height="500" style="display: block; margin: auto;" />

You can control the player using the keyboard, just like in the original Minecraft game.
```{hint}
We also provide some function keys to help you control the game.
`C` is to capture or release the system cursor.
`Left Ctrl + C` is to exit the game (and shut down the simulator).
`Esc` is to enter *Command Mode* (read more in the next section).
```

````{dropdown} <i class="fa-solid fa-lightbulb" height="35px" width="20px"></i> Learn more about Minecraft keyboard controls
| Key                  | Action                                   |
| -------------------- | ---------------------------------------- |
| `W`                  | Move forward                             |
| `A`                  | Move left                                |
| `S`                  | Move backward                            |
| `D`                  | Move right                               |
| `Space`              | Jump                                     |
| `Shift`              | Sneak                                    |
| `E`                  | Open inventory                           |
| `Q`                  | Drop item                                |
| `1-9`                | Select hotbar slot                       |
| `F`                  | Swap item in hand with item in inventory |
| `Left mouse button`  | Destroy block                            |
| `Right mouse button` | Place block                              |
| `Mouse movement`     | Change view direction                    |
| `Mouse wheel`        | Change hotbar slot                       |
````

## Command Mode

We allow users to define custom functions for various usage scenarios.
By using Callbacks, you can define your own functions and bind them to specific keys on the keyboard.
Also, you can implement your own `DrawCall` function and pass it to the `extra_draw_call` parameter in the `PlayCallback` to draw custom information on the screen.
Press the Esc key to enter Command Mode, where you can view the available functions and their corresponding keys.
When you press a key, the associated callback function will be enabled or disabled accordingly.
Below is an image of Command Mode:
<img src="../_static/image/command_mode.png" alt="Alt text" height="500" style="display: block; margin: auto;" />

We have implemented some built-in functions or callbacks for you to use. You can find an example below:
```python
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback, PlaySegmentCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall, SegmentDrawCall
from functools import partial
from minestudio.models import load_rocket_policy
if __name__ == '__main__':
    agent_generator = partial(
        load_rocket_policy,
        ckpt_path = # your checkpoint path
    )
    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            PlaySegmentCallback(sam_path='./minestudio/models/realtime_sam/checkpoints', sam_choice='small'),
            PlayCallback(agent_generator=agent_generator, extra_draw_call=[RecordDrawCall, CommandModeDrawCall, SegmentDrawCall]),
            RecordCallback(record_path='./output', recording=False),
        ]
    )
    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()
```
The `PlayCallback` is modded by three extra functions or callbacks and `CommandModeDrawCall` is used to draw the command mode on the screen.

When `agent_generator` yields a usable policy for `PlayCallback`, you will be able to switch between the agent and human control by pressing the `L` key under the Command Mode.

The `RecordCallback` is used to record the video by pressing the `R` key under the Command Mode and press `R` again to stop recording.
`RecordDrawCall` is used to draw the recording status on the screen when recording:
<img src="../_static/image/record.png" alt="Alt text" height="500" style="display: block; margin: auto;" />
You can find this function useful when recording human demonstrations or evaluating policies.

The `PlaySegmentCallback` and `SegmentDrawCall` are useful when you want to segment an object in the scene (for tasks such as data labeling or providing instructions to the agent).
Press the `S` key in Command Mode to initiate the segmentation process using the Segment Anything Model (SAM).
A window will appear, allowing you to left-click to select the object you want to segment and right-click to add negative samples.
Press `C` to clear the selection, and press `Enter` to start tracking the object.
To stop tracking, press `Esc` to exit the segmentation process without tracking.
If an object is being tracked, `SegmentDrawCall` will draw the mask of the object on the screen in real-time.
Press `S` again to stop tracking in command mode.
Here is an example of the segmentation and tracking process:

<div style="text-align: center;">
  <p style="display: inline-block; margin-right: 20px;">
    <img src="../_static/image/segmenting.png" alt="Alt text" height="400" />
    <br />
    <em>Segmentation Labeling</em>
  </p>
  <p style="display: inline-block;">
    <img src="../_static/image/tracking.png" alt="Alt text" height="400" />
    <br />
    <em>Tracking Object in real-time</em>
  </p>
</div>


```{warning}
Be sure to download Segment Anyting Model (SAM) by running `minestudio/models/realtime_sam/checkpoints/download_ckpts.sh`.
You can choose model size by setting `sam_choice` to `tiny`, `small`, `base`, or `large`.
The `PlaySegmentCallback` should always be placed before the `PlayCallback` in the callback list to avoid conflicts.
```