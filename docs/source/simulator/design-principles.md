# Design Principles

## Simulator Lifecycle and Callback Integration

The MineStudio simulator follows a standard reinforcement learning environment lifecycle, including `reset`, `step`, `render`, and `close` methods. A key design principle is the integration of a flexible callback system, allowing users to hook into these lifecycle methods to customize behavior without modifying the core simulator code.

Callbacks are executed in the order they are provided in the `callbacks` list during `MinecraftSim` initialization.

- **`reset()`**: Initializes or resets the environment to a starting state.
    - **`before_reset(self, sim, reset_flag: bool) -> bool`**: Executed for each callback before the main reset logic. It receives the simulator instance (`sim`) and a `reset_flag`. A callback can return `False` to potentially suppress the underlying `self.env.reset()` call (e.g., for a custom fast reset). The `reset_flag` passed to subsequent callbacks is the result of the previous one.
    - The core environment reset (`self.env.reset()`) is called if `reset_flag` remains `True` after all `before_reset` calls.
    - A fixed number of no-op actions (`self.num_empty_frames`) are then performed to skip initial loading frames.
    - The observation and info are wrapped by `_wrap_obs_info`.
    - **`after_reset(self, sim, obs, info)`**: Executed for each callback after the main reset logic and initial frame skipping. It receives the simulator instance, the initial observation (`obs`), and info dictionary (`info`). Callbacks can modify `obs` and `info` here. The modified `obs` and `info` are passed to subsequent callbacks.
    - The final `obs` and `info` are returned.

    ```python
    # Simplified structure of MinecraftSim.reset()
    def reset(self) -> Tuple[np.ndarray, Dict]:
        reset_flag = True
        for callback in self.callbacks:
            reset_flag = callback.before_reset(self, reset_flag) # Hook before reset
        
        if reset_flag: # Main environment reset
           self.env.reset()
           self.already_reset = True
        
        for _ in range(self.num_empty_frames): # Skip initial frames
            action = self.env.action_space.no_op()
            obs, reward, done, info = self.env.step(action)
        
        obs, info = self._wrap_obs_info(obs, info) # Wrap observation and info
        
        for callback in self.callbacks:
            # Hook after reset, can modify obs and info
            obs, info = callback.after_reset(self, obs, info) 
            self.obs, self.info = obs, info # Update internal state
        return obs, info
    ```

    ```{hint}
    **Use Cases for `reset` callbacks:**
    *   **Custom Initialization**: Use `after_reset` to send commands (e.g., `/time set day`, `/give`), set player properties, or log initial state.
    *   **Fast Reset**: Implement `before_reset` to return `False` and handle resetting the agent's state (e.g., teleport, clear inventory) without a full environment reload. `after_reset` can then finalize this custom reset.
    *   **Observation/Info Augmentation**: Add task-specific information or modify the initial observation in `after_reset`.
    ```

- **`step(action)`**: Executes one time-step in the environment.
    - If `action_type` is `'agent'`, the input `action` is first converted to the environment's action format using `agent_action_to_env_action`.
    - **`before_step(self, sim, action)`**: Executed for each callback. It receives the simulator instance and the (potentially converted) `action`. Callbacks can modify the `action` before it's passed to the environment. The modified `action` is passed to subsequent callbacks.
    - The core environment step (`self.env.step(action.copy())`) is performed.
    - `terminated` and `truncated` flags are set (both to `done` in the current implementation).
    - The observation and info are wrapped by `_wrap_obs_info`.
    - **`after_step(self, sim, obs, reward, terminated, truncated, info)`**: Executed for each callback. It receives the simulator instance and the results from `self.env.step()`. Callbacks can modify these values. The modified values are passed to subsequent callbacks.
    - The final `obs`, `reward`, `terminated`, `truncated`, and `info` are returned.

    ```python
    # Simplified structure of MinecraftSim.step()
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.action_type == 'agent':
            env_action = self.agent_action_to_env_action(action)
            # ... action dictionary manipulation ...
            action.update(env_action)
            
        for callback in self.callbacks:
            action = callback.before_step(self, action) # Hook before step

        obs, reward, done, info = self.env.step(action.copy()) # Main environment step

        terminated, truncated = done, done # Determine termination
        obs, info = self._wrap_obs_info(obs, info) # Wrap observation and info
        
        for callback in self.callbacks:
            # Hook after step, can modify results
            obs, reward, terminated, truncated, info = callback.after_step(self, obs, reward, terminated, truncated, info)
            self.obs, self.info = obs, info # Update internal state
        return obs, reward, terminated, truncated, info
    ```

    ```{hint}
    **Use Cases for `step` callbacks:**
    *   **Action Masking/Modification**: Change or restrict actions in `before_step`.
    *   **Custom Reward Shaping**: Modify the `reward` in `after_step` based on `obs` or `info`.
    *   **Trajectory Recording**: Log `obs`, `action`, `reward`, `info` in `after_step`.
    *   **Early Termination**: Modify `terminated` or `truncated` flags in `after_step` based on custom conditions.
    ```

- **`render()`**: Renders the current environment observation.
    - Retrieves the current observation image (`self.obs['image']`).
    - **`before_render(self, sim, image)`**: Executed for each callback. Receives the simulator instance and the current `image`. Callbacks can modify the `image` (e.g., add overlays, annotations) before the main rendering logic (if any) or before it's passed to subsequent callbacks.
    - **`after_render(self, sim, image)`**: Executed for each callback. Receives the simulator instance and the (potentially modified by `before_render`) `image`. Callbacks can further process the `image`.
    - The final `image` is returned.

    ```python
    # Structure of MinecraftSim.render()
    def render(self) -> None:
        image = self.obs['image']
        for callback in self.callbacks:
            image = callback.before_render(self, image) # Hook before rendering modifications
        
        # ! core logic (currently, core logic is minimal, focus is on callbacks)
        
        for callback in self.callbacks:
            image = callback.after_render(self, image) # Hook after rendering modifications
        return image
    ```

    ```{hint}
    **Use Cases for `render` callbacks:**
    *   **Visualization Augmentation**: Use `before_render` or `after_render` to draw debug information, agent stats, or highlight important elements on the frame.
    *   **Image Preprocessing for Display**: Resize or format the image for specific display requirements.
    ```

- **`close()`**: Cleans up and closes the environment.
    - **`before_close(self, sim)`**: Executed for each callback before the underlying environment is closed. Useful for saving final data or logs.
    - The core environment close (`self.env.close()`) is called.
    - **`after_close(self, sim)`**: Executed for each callback after the underlying environment has been closed. Useful for final cleanup tasks that depend on the environment being closed.
    - The status from `self.env.close()` is returned.

    ```python
    # Structure of MinecraftSim.close()
    def close(self) -> None:
        for callback in self.callbacks:
            callback.before_close(self) # Hook before closing
        close_status = self.env.close() # Main environment close
        for callback in self.callbacks:
            callback.after_close(self) # Hook after closing
        return close_status
    ```

    ```{hint}
    **Use Cases for `close` callbacks:**
    *   **Final Data Saving**: Save recorded trajectories, statistics, or model checkpoints in `before_close`.
    *   **Resource Release**: Release any resources acquired by callbacks during the simulation.
    ```

## Callbacks Base Class

Callbacks are classes that inherit from `MinecraftCallback` and can override any of its methods to inject custom logic at different points in the simulator's lifecycle. All callback methods receive the simulator instance (`sim`) as their first argument, allowing them to access and potentially modify the simulator's state or data.

The base `MinecraftCallback` class defines the following methods, all of which simply pass through the data by default:

```python
class MinecraftCallback:
    
    def before_step(self, sim, action):
        """Called before `env.step()`.
        
        Args:
            sim: The MinecraftSim instance.
            action: The action to be taken.
        Returns:
            The potentially modified action.
        """
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """Called after `env.step()`.
        
        Args:
            sim: The MinecraftSim instance.
            obs: The observation from the environment.
            reward: The reward from the environment.
            terminated: The terminated flag from the environment.
            truncated: The truncated flag from the environment.
            info: The info dictionary from the environment.
        Returns:
            A tuple of (obs, reward, terminated, truncated, info), potentially modified.
        """
        return obs, reward, terminated, truncated, info
    
    def before_reset(self, sim, reset_flag: bool) -> bool:
        """Called before `env.reset()`.
        
        Args:
            sim: The MinecraftSim instance.
            reset_flag: Boolean indicating if a hard reset should occur.
        Returns:
            Boolean indicating if the hard reset should still occur.
        """
        return reset_flag
    
    def after_reset(self, sim, obs, info):
        """Called after `env.reset()` and initial frame skipping.
        
        Args:
            sim: The MinecraftSim instance.
            obs: The initial observation.
            info: The initial info dictionary.
        Returns:
            A tuple of (obs, info), potentially modified.
        """
        return obs, info
    
    def before_close(self, sim):
        """Called before `env.close()`.
        
        Args:
            sim: The MinecraftSim instance.
        """
        return
    
    def after_close(self, sim):
        """Called after `env.close()`.
        
        Args:
            sim: The MinecraftSim instance.
        """
        return
    
    def before_render(self, sim, image):
        """Called before the main rendering logic in `sim.render()`.
        
        Args:
            sim: The MinecraftSim instance.
            image: The current image to be rendered.
        Returns:
            The potentially modified image.
        """
        return image
    
    def after_render(self, sim, image):
        """Called after the main rendering logic in `sim.render()`.
        
        Args:
            sim: The MinecraftSim instance.
            image: The image after initial rendering/modifications.
        Returns:
            The potentially modified image.
        """
        return image
```
By implementing one or more of these methods in a custom callback class, users can precisely control and extend the behavior of the MineStudio simulator.