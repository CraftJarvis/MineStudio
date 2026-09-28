'''
Date: 2024-11-14 20:10:54
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-20 01:06:35
FilePath: /MineStudio/minestudio/simulator/callbacks/play.py
'''
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator.utils import MinecraftGUI, GUIConstants

import time
from typing import Dict, Literal, Optional, Callable, Tuple, List, Any
from minestudio.utils import get_compute_device
from rich import print

DEBUG = False

class PlayCallback(MinecraftCallback):
    """Enables interactive play and/or agent-driven gameplay in a GUI window.

    This callback provides a graphical interface for human players to interact
    with the Minecraft environment. It can also run a pre-trained agent and
    allow switching control between human and agent.

    Key functionalities:
    - Renders the game view in a separate window.
    - Captures keyboard and mouse input for human control.
    - Can load and run a specified agent model.
    - Allows switching between human and agent control (default: 'L' key).
    - Displays game information (FPS, player position, current mode) in the GUI.
    - Handles custom key bindings for actions like mouse capture and closing.

    :param agent_generator: A callable that returns an agent instance. 
                            If None, only human play is enabled. Defaults to None.
    :type agent_generator: Callable, optional
    :param extra_draw_call: A list of additional callable functions to be executed
                              during the GUI drawing phase. Defaults to None.
    :type extra_draw_call: Optional[List[Callable]], optional
    """
    def __init__(
        self,
        agent_generator: Callable = None,
        extra_draw_call: Optional[List[Callable]] = None
    ):
        """Initializes the PlayCallback.

        Sets up the GUI, loads the agent if provided, and prints key bindings.

        :param agent_generator: Function to generate the agent model.
        :param extra_draw_call: Additional functions for custom GUI drawing.
        """
        self.gui = MinecraftGUI(extra_draw_call=extra_draw_call)
        self.constants = GUIConstants()
        self.start_time = time.time()
        self.end_time = time.time()
        if agent_generator is not None:
            print(f'[green]Load agent with name: {agent_generator.func.__name__}, args: {agent_generator.keywords}[/green]')
            self.agent = agent_generator().to(get_compute_device())
        else:
            self.agent = None
        self.switch = 'human'
        self.terminated = False
        self.last_action = None
        self.timestep = 0

        if self.agent is not None:
            self.reset_agent()

        # print a table of key bindings
        print(
            f'[yellow]Extra Key Bindings Besides Minecraft Controls:[/yellow]\n'
            f'  [white]C[/white]: Capture Mouse \n'
            f'  [white]Left Ctrl + C[/white]: [red]Close Window[/red] \n'
            f'  [white]Esc[/white]: [green]Enter Command Mode[/green] \n'
        )
    
    def reset_agent(self):
        """Resets the agent's internal state (e.g., memory).

        Called when switching to agent control or at the start of a new episode.
        """
        self.memory = None

    def before_reset(self, sim, reset_flag: bool) -> bool:
        """Resets the GUI before the environment resets.

        :param sim: The simulator instance.
        :param reset_flag: The current reset flag status.
        :returns: The passed `reset_flag`.
        :rtype: bool
        """
        self.gui.reset_gui()
        return reset_flag
    
    def after_reset(self, sim, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
        """Handles tasks after the environment resets.

        Resets termination flag, agent state, updates GUI, and resets timestep.

        :param sim: The simulator instance.
        :param obs: The initial observation.
        :param info: The initial info dictionary.
        :returns: The passed `obs` and `info`.
        :rtype: Tuple[Dict, Dict]
        """
        self.terminated = False
        if self.agent is not None:
            self.reset_agent()
        sim.callback_messages.add("Press 'L' to switch control.")
        self.gui._update_image(info)
        self.timestep = 0
        return obs, info
    
    def before_step(self, sim, action: Any) -> Dict:
        """Determines and processes the action before the environment steps.

        Handles input from human (keyboard/mouse via GUI) or agent based on
        the current control switch (`self.switch`). Also processes chat messages.

        The executed action will be added to the info dict as "taken_action".

        :param sim: The simulator instance.
        :param action: The proposed action (can be None, a string like 'human', or an action dict).
        :type action: Any
        :returns: The action dictionary to be executed by the environment.
        :rtype: Dict
        """
        assert not self.terminated, "Cannot step environment after it is done."

        self.gui.window.dispatch_events()
        if isinstance(action, str) or action is None:
            if isinstance(action, str) and action != self.switch:
                self.switch = action
                if self.switch == "agent":
                    if self.agent is None:
                        print('[red]agent is not specified, switch to human control[/red]')
                        self.switch = 'human'
                    else:
                        self.reset_agent()

            if self.switch != "agent":
                if self.gui.command != "":
                    action = sim.noop_action()
                else:
                    human_action = self.gui._get_human_action()
                    action = human_action
            else:
                assert self.agent is not None, "Agent is not specified."
                agent_action, self.memory = self.agent.get_action(sim.obs, self.memory, input_shape = "*")
                agent_action = sim.agent_action_to_env_action(agent_action)
                action = agent_action
        
        if self.gui.chat_message is not None: #!WARNING: should stop the game when chat message is not None 
            action["chat"] = self.gui.chat_message
            self.gui.chat_message = None

        self.last_action = action
        return action
    
    def after_step(self, sim, obs: Dict, reward: float, terminated: bool, truncated: bool, info: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Handles tasks after the environment takes a step.

        Updates GUI with new observation and info, calculates FPS, processes key releases
        (like switching control or mouse capture), and handles termination.

        :param sim: The simulator instance.
        :param obs: The observation after the step.
        :param reward: The reward received.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: The info dictionary.
        :returns: The (potentially modified) obs, reward, terminated, truncated, and info.
        :rtype: Tuple[Dict, float, bool, bool, Dict]
        """
        self.terminated = terminated
        self.timestep += 1
        self.end_time = time.time()
        time.sleep(max(0, self.constants.MINERL_FRAME_TIME - (self.end_time - self.start_time)))
        fps = 1 / (self.end_time - self.start_time)
        self.start_time = time.time()

        released_keys = self.gui._capture_all_keys()

        if 'ESCAPE' in released_keys:
            info['ESCAPE'] = True
            self.gui.mode = 'command'
            print(f'[green]Command Mode Activated[/green]')
        message = [
            [f"Role: {self.switch}", f"Mode: {self.gui.mode}", f"Timestep: {self.timestep}", f"FPS: {fps:.2f}"], 
            [f"X: {info['player_pos']['x']:.2f}", f"Y: {info['player_pos']['y']:.2f}", f"Z: {info['player_pos']['z']:.2f}"],
        ]

        if DEBUG:
            print(f'[yellow]Debug Information:[/yellow]')
            ignored_keys = ['location_stats', 'voxels', 'mobs', 'health', 'food_level', 'pov', 'inventory', 'equipped_items', 'use_item', 'pickup', 'break_item', 'craft_item', 'mine_block', 'damage_dealt', 'kill_entity', 'player_pos', 'is_gui_open', 'isGuiOpen']
            for k, v in info.items():
                if k not in ignored_keys:
                    print(f'{k}: {v}')
            print(f'[End of Debug Information]')

        for name, message_item in info.get('message', {}).items():
            message.append([message_item])

        action_items = []
        for k, v in self.last_action.items():
            if k == 'camera':
                v = f"({v[0]:.2f}, {v[1]:.2f})"
            elif 'hotbar' in k:
                continue
            elif 'chat' in k:
                continue
            action_items.append(f"{k}: {v}")
        message.append(action_items)
         
        self.gui._update_image(info, message=message)
        info['ESCAPE'] = False # don't forget to reset the key

        if self.gui.mode == 'command':
            help_message = ""
            for message_item in sim.callback_messages:
                help_message += message_item + '\n'
            self.gui._show_message(help_message)

        released_keys = self.process_keys(sim, released_keys)

        for key in released_keys:
            if key in info:
                info[key] = not info[key]
            else:
                info[key] = True

        terminated = self.terminated

        # press 'L' to switch control
        if 'L' in released_keys:
            switch_control = True
            self.switch = 'human' if self.switch == 'agent' else 'agent'
            print(f'[red]Switch to {self.switch} control[/red]')
        else:
            switch_control = False

        if terminated:
            self.gui._show_message("Episode terminated.")

        info["taken_action"] = self.last_action
        info['switch'] = self.switch
        self.terminated = terminated

        if switch_control:
            #? TODO: add more features after switching control
            if self.switch == 'agent':
                if self.agent is None:
                    print('[red]agent is not specified, switch to human control[/red]')
                    self.switch = 'human'
                else:
                    self.reset_agent()
            obs, reward, terminated, truncated, info = sim.step(sim.noop_action())
            self.last_action = sim.noop_action()
            self.terminated = terminated
            self.timestep += 1

        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        """Closes the GUI window before the simulator closes.

        :param sim: The simulator instance.
        """
        self.gui.close_gui()

    def process_keys(self, sim, released_keys: set) -> set:
        """Processes special key releases for GUI and simulation control.

        Handles:
        - 'C': Toggle mouse capture (exclusive mouse mode).
        - Ctrl+'C': Close the window and terminate the simulation.
        - 'ESCAPE': Enter/exit command mode (currently exits by clearing keys).

        :param sim: The simulator instance.
        :param released_keys: A set of keys that were released in this frame.
        :type released_keys: set
        :returns: The set of `released_keys` after processing (potentially modified).
        :rtype: set
        """
        # press 'C' to set mouse visibility
        if 'C' in released_keys:
            # print('shit')
            if not (self.gui.modifiers & self.gui.key.MOD_CTRL):
                self.gui.capture_mouse = not self.gui.capture_mouse
                self.gui.window.set_mouse_visible(not self.gui.capture_mouse)
                self.gui.window.set_exclusive_mouse(self.gui.capture_mouse)
            else:
                # press ctrl+C to close the window and stop the simulation
                print(f'[red]Close the window![/red]')
                self.terminated = True

        # press 'ESC' to enter command mode
        if 'ESCAPE' in released_keys:
            time_count = 0 # Renamed variable to avoid conflict with time module
            while True:
                self.gui.window.dispatch_events()
                self.gui.window.switch_to()
                self.gui.window.flip()
                current_released_keys = self.gui._capture_all_keys() # Use a different variable name
                time_count += 1
                if len(current_released_keys) > 0:
                    released_keys = current_released_keys # Update the original set if needed
                    break
            self.gui.mode = 'normal'
            # delete ESCAPE in released keys
            released_keys = set(released_keys) - {'ESCAPE'}
            if 'C' in released_keys and (self.gui.modifiers & self.gui.key.MOD_CTRL):
                print(f'[red]Close the window![/red]')
                self.terminated = True
        else:
            # delete all keys in the released_keys
            released_keys = set()
        
        return released_keys


