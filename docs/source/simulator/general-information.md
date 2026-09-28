# General Information


## Observation Space
Most environments use the same observation space (just an RGB image), refer to [MineRL](https://minerl.readthedocs.io/en/latest/environments/index.html#observation-space).

```python
from minestudio.simulator import MinecraftSim
sim = MinecraftSim(obs_size=(224, 224), render_size=(640, 360))
obs, info = sim.reset()
```
The observation space is a dictionary with the following keys:
```python
Dict({
    "image": Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
})
```
```{note}
Here are two types of resolution: ``render_size`` and ``obs_size``. The ``render_size`` specifies the resolution of the Minecraft window, usually set to {math}`640 \times 360`. We use opencv ``resize`` function to resize the image to the ``obs_size``. 

You can also access the original image in ``info["pov"]`` returned by the ``step`` function or the ``reset`` function.  
```

```{warning}
We use ``cv2.INTER_LINEAR`` as the interpolation method when resizing the image. 

We found that it is very important to align the interpolation methods between training and inference; otherwise, it may lead to poor performance. 
```

## Action Space

We provide two types of action spaces: ``env`` and ``agent``. 

```{note}
``Discrete`` and ``Box`` are from the ``gymnasium.spaces`` module. 
```

- ``env``: The action space is similar to the original MineRL environment. 


    ```python
    Dict({
        "attack": Discrete(2),
        "back": Discrete(2),
        "camera": Box(low=-180.0, high=180.0, shape=(2,)),
        "forward": Discrete(2),
        "hotbar.1": Discrete(2),
        "hotbar.2": Discrete(2),
        "hotbar.3": Discrete(2),
        "hotbar.4": Discrete(2),
        "hotbar.5": Discrete(2),
        "hotbar.6": Discrete(2),
        "hotbar.7": Discrete(2),
        "hotbar.8": Discrete(2),
        "hotbar.9": Discrete(2),
        "inventory": Discrete(2),
        "jump": Discrete(2),
        "left": Discrete(2),
        "right": Discrete(2),
        "sneak": Discrete(2),
        "sprint": Discrete(2),
        "use": Discrete(2)"
    })
    ```

    ```{note}
    This kind of action space is human-friendly. 
    ```

- ``agent``: The action space is a dictionary with the following keys:

    ```python
    Dict({
        "buttons": MultiDiscrete([8641]),
        "camera":  MultiDiscrete([121])
    })
    ```

    ```{note}
    It adopts the hierarchical design principle from the [video-pretraining](https://github.com/openai/Video-Pre-Training) paper. 

    This kind of action space is more suitable for training polices. 
    ```


## Information

The information space is a dictionary with the following keys:

```python
dict_keys([
    'isGuiOpen', 'location_stats', 'voxels', 'mobs', 'health', 
    'food_level', 'pov', 'inventory', 'equipped_items', 'use_item'
    'pickup', 'break_item', 'craft_item', 'mine_block', 'damage_dealt', 
    'kill_entity', 'player_pos', 'is_gui_open', 'message'
])
```

```{hint}
The information is recorded in an accumulated manner, for example, ``kill entity`` records the cumulative number of entities killed. 
```