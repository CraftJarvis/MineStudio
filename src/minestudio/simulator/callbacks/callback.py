'''
Date: 2025-01-06 17:32:04
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-06-15 17:02:18
FilePath: /MineStudio/minestudio/simulator/callbacks/callback.py
'''
import os
import yaml
import random
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
class MinecraftCallback:
    """
    Base class for creating callbacks that can be used to customize the behavior of the Minecraft simulator.
    Callbacks can be used to modify actions, observations, rewards, and other aspects of the simulation.
    """
    
    def load_data_from_conf(source: Union[str, Dict]) -> Dict:
        """
        Loads data from a YAML file or a dictionary.

        :param source: The path to the YAML file or a dictionary containing the data.
        :type source: Union[str, Dict]
        :raises AssertionError: if the file specified by `source` does not exist.
        :returns: A dictionary containing the loaded data.
        :rtype: Dict
        """
        if isinstance(source, Dict):
            data = source
        else:
            assert os.path.exists(source), f"File {source} not exists."
            with open(source, 'r') as f:
                data = yaml.safe_load(f)
        return data

    def create_from_conf(yaml_file: Union[str, Dict]):
        """
        Creates a callback instance from a YAML file or a dictionary.
        This method is intended to be overridden by subclasses.

        :param yaml_file: The path to the YAML file or a dictionary containing the configuration.
        :type yaml_file: Union[str, Dict]
        :returns: A callback instance, or None if not implemented.
        :rtype: Optional[MinecraftCallback]
        """
        return None

    def before_step(self, sim, action):
        """
        Called before the simulator takes a step.
        This method can be used to modify the action before it is executed.

        :param sim: The simulator instance.
        :type sim: any
        :param action: The action to be executed.
        :type action: any
        :returns: The (potentially modified) action.
        :rtype: any
        """
        return action

    def before_reset(self, sim, reset_flag: bool) -> bool: # whether need to call env reset
        """
        Called before the simulator resets the environment.
        This method can be used to determine if the environment needs to be reset.

        :param sim: The simulator instance.
        :type sim: any
        :param reset_flag: A boolean flag indicating whether a reset is currently planned.
        :type reset_flag: bool
        :returns: A boolean flag indicating whether the environment should be reset.
        :rtype: bool
        """
        return reset_flag
    
    def after_reset(self, sim, obs, info):
        """
        Called after the simulator resets the environment.
        This method can be used to modify the initial observation and info.

        :param sim: The simulator instance.
        :type sim: any
        :param obs: The initial observation after reset.
        :type obs: any
        :param info: The initial info dictionary after reset.
        :type info: Dict
        :returns: The (potentially modified) observation and info.
        :rtype: Tuple[Any, Dict]
        """
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """
        Called after the simulator takes a step.
        This method can be used to modify the observation, reward, done flags, and info.

        :param sim: The simulator instance.
        :type sim: any
        :param obs: The observation after the step.
        :type obs: any
        :param reward: The reward received after the step.
        :type reward: float
        :param terminated: A boolean flag indicating if the episode has terminated.
        :type terminated: bool
        :param truncated: A boolean flag indicating if the episode has been truncated.
        :type truncated: bool
        :param info: The info dictionary after the step.
        :type info: Dict
        :returns: The (potentially modified) observation, reward, terminated flag, truncated flag, and info.
        :rtype: Tuple[Any, float, bool, bool, Dict]
        """
        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        """
        Called before the simulator is closed.
        This method can be used to perform any cleanup tasks.

        :param sim: The simulator instance.
        :type sim: any
        """
        pass

    def after_close(self, sim):
        """
        Called after the simulator is closed.
        This method can be used to perform final cleanup operations.

        :param sim: The simulator instance.
        :type sim: any
        """
        return

    def before_render(self, sim, image):
        """
        Called before the simulator renders an image.
        This method can be used to modify the image before it is rendered.

        :param sim: The simulator instance.
        :param image: The image to be rendered.
        :returns: The (potentially modified) image.
        """
        return image

    def after_render(self, sim, image):
        """
        Called after the simulator renders an image.
        This method can be used to modify the image after it is rendered.

        :param sim: The simulator instance.
        :param image: The rendered image.
        :returns: The (potentially modified) image.
        """
        return image

    def __repr__(self):
        return f"{self.__class__.__name__}()"
class Compose(MinecraftCallback):
    """
    A callback that composes multiple callbacks into a single callback.
    It allows for applying a sequence of callbacks or a random subset of them.
    """
    
    def __init__(self, callbacks:list, options:int=-1):
        """
        Initializes the Compose callback.

        :param callbacks: A list of MinecraftCallback instances to compose.
        :type callbacks: list
        :param options: The number of callbacks to randomly select and activate from the list. 
                        If -1, all callbacks are activated. If 0, no callbacks are activated.
                        Defaults to -1.
        :type options: int
        :raises AssertionError: if `options` is not within the valid range [0, len(callbacks)] when not -1.
        """
        self.callbacks = callbacks
        self.options = options
        self.activate_callbacks = []

    def before_reset(self, sim, reset_flag: bool) -> bool:
        """
        Called before the simulator resets the environment.
        Activates a subset of callbacks based on the `options` attribute and calls their `before_reset` methods.

        :param sim: The simulator instance.
        :param reset_flag: A boolean flag indicating whether a reset is currently planned.
        :type reset_flag: bool
        :returns: A boolean flag indicating whether the environment should be reset, after processing by active callbacks.
        :rtype: bool
        """
        if self.options == -1:
            self.activate_callbacks = self.callbacks
        else:
            assert 0 <= self.options <= len(self.callbacks), f"{self.options}"
            self.activate_callbacks = random.sample(self.callbacks, k=self.options)
        for callback in self.activate_callbacks:
            reset_flag = callback.before_reset(sim, reset_flag)
        return reset_flag

    def after_reset(self, sim, obs, info):
        """
        Called after the simulator resets the environment.
        Calls the `after_reset` method of all activated callbacks.

        :param sim: The simulator instance.
        :param obs: The initial observation after reset.
        :param info: The initial info dictionary after reset.
        :returns: The (potentially modified) observation and info, after processing by active callbacks.
        :rtype: Tuple[Any, Dict]
        """
        for callback in self.activate_callbacks:
            obs, info = callback.after_reset(sim, obs, info)
        return obs, info

    def before_step(self, sim, action):
        """
        Called before the simulator takes a step.
        Calls the `before_step` method of all activated callbacks.

        :param sim: The simulator instance.
        :param action: The action to be executed.
        :returns: The (potentially modified) action, after processing by active callbacks.
        """
        for callback in self.activate_callbacks:
            action = callback.before_step(sim, action)
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """
        Called after the simulator takes a step.
        Calls the `after_step` method of all activated callbacks.

        :param sim: The simulator instance.
        :param obs: The observation after the step.
        :param reward: The reward received after the step.
        :param terminated: A boolean flag indicating if the episode has terminated.
        :param truncated: A boolean flag indicating if the episode has been truncated.
        :param info: The info dictionary after the step.
        :returns: The (potentially modified) observation, reward, terminated flag, truncated flag, and info, after processing by active callbacks.
        :rtype: Tuple[Any, float, bool, bool, Dict]
        """
        for callback in self.activate_callbacks:
            obs, reward, terminated, truncated, info = callback.after_step(sim, obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        """
        Called before the simulator is closed.
        Calls the `before_close` method of all activated callbacks.

        :param sim: The simulator instance.
        """
        for callback in self.activate_callbacks:
            callback.before_close(sim)
        return

    def after_close(self, sim):
        """
        Called after the simulator is closed.
        Calls the `after_close` method of all activated callbacks.

        :param sim: The simulator instance.
        """
        for callback in self.activate_callbacks:
            callback.after_close(sim)
        return

    def before_render(self, sim, image):
        """
        Called before the simulator renders an image.
        Calls the `before_render` method of all activated callbacks.

        :param sim: The simulator instance.
        :param image: The image to be rendered.
        :returns: The (potentially modified) image, after processing by active callbacks.
        """
        for callback in self.activate_callbacks:
            image = callback.before_render(sim, image)
        return image

    def after_render(self, sim, image):
        """
        Called after the simulator renders an image.
        Calls the `after_render` method of all activated callbacks. 
        Note: This method currently calls `before_render` on the callbacks, which might be an error. It should likely call `after_render`.

        :param sim: The simulator instance.
        :param image: The rendered image.
        :returns: The (potentially modified) image, after processing by active callbacks.
        """
        for callback in self.activate_callbacks:
            image = callback.before_render(sim, image)
        return image
    
