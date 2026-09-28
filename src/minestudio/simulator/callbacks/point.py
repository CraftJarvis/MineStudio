'''
Date: 2024-11-18 20:37:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:23:45
FilePath: /MineStudio/minestudio/simulator/callbacks/point.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator.utils import MinecraftGUI, GUIConstants
from minestudio.simulator.utils.gui import PointDrawCall, SegmentDrawCall, MultiPointDrawCall

import time
from typing import Dict, Literal, Optional, Callable, Tuple, List, Any
from rich import print
import numpy as np
import cv2
import os


class PointCallback(MinecraftCallback):
    """Allows the player to select a point on the screen using the mouse.

    When activated (default: by pressing 'P'), this callback opens a GUI window
    displaying the current game view. The player can click on this window to
    select a 2D point. The selected point's coordinates are stored in the
    `info['point']` dictionary.
    """
    def __init__(self):
        """Initializes the PointCallback."""
        super().__init__()
        
    def after_reset(self, sim, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
        """Adds a message to the simulator to inform the user about the activation key.

        :param sim: The simulator instance.
        :param obs: The initial observation.
        :param info: The initial info dictionary.
        :returns: The passed `obs` and `info`.
        :rtype: Tuple[Dict, Dict]
        """
        sim.callback_messages.add("Press 'P' to start pointing.")
        return obs, info

    def after_step(self, sim, obs: Dict, reward: float, terminated: bool, truncated: bool, info: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Handles the point selection process if activated.

        If the 'P' key is pressed (i.e., `info.get('P', False)` is True):
        1. Opens a new GUI window.
        2. Enters a loop to capture mouse clicks for point selection.
        3. Updates `info['point']` with the clicked coordinates.
        4. Closes the GUI when 'ESCAPE' is pressed.

        :param sim: The simulator instance.
        :param obs: The observation after the step.
        :param reward: The reward received.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: The info dictionary.
        :returns: The (potentially modified) obs, reward, terminated, truncated, and info.
        :rtype: Tuple[Dict, float, bool, bool, Dict]
        """
        if info.get('P', False):
            print(f'[green]Start pointing[/green]')
        else:
            return obs, reward, terminated, truncated, info
        
        gui = MinecraftGUI(extra_draw_call=[PointDrawCall], show_info=False)
        gui.window.activate()

        while True:
            gui.window.dispatch_events()
            gui.window.switch_to()
            gui.window.set_mouse_visible(True)
            gui.window.set_exclusive_mouse(False)
            gui.window.flip()
            released_keys = gui._capture_all_keys()
            if 'ESCAPE' in released_keys:
                break
            if gui.mouse_position is not None:
                info['point'] = gui.mouse_position
            gui._show_image(info)
            
        gui.close_gui()

        if info['point'] is not None:
            print(f'[red]Stop pointing at {info["point"]}[/red]')
        info['P'] = False
        return obs, reward, terminated, truncated, info
        
class PlaySegmentCallback(MinecraftCallback):
    """Integrates Segment Anything Model (SAM) for interactive object segmentation.

    This callback allows a human player to provide positive and negative point prompts
    on the game's POV to segment objects using a SAM2 model. It then tracks the
    segmented object across subsequent frames.

    **Note:** This callback should typically be placed *before* the `PlayCallback`
    in the callback list to ensure its GUI interactions are handled correctly.

    Key Features:
    - Loads a specified SAM2 model checkpoint.
    - Provides a GUI for adding positive/negative point prompts.
    - Generates an initial segmentation based on prompts.
    - Tracks the segmented object in subsequent frames.
    - Adds the segmentation mask to `obs['segment']['obj_mask']`.

    Activation:
    - Press 'S' to start/stop the segmentation process.

    GUI Controls (during segmentation):
    - Left Mouse Click: Add a positive point prompt.
    - Right Mouse Click: Add a negative point prompt.
    - 'C': Clear all current points.
    - 'Enter': Start tracking the current segmentation.
    - 'ESCAPE': Exit segmentation mode.

    :param sam_path: Path to the directory containing SAM2 model checkpoints and configs.
    :type sam_path: str
    :param sam_choice: Which SAM2 model to load (e.g., 'base', 'large', 'small', 'tiny').
                       Defaults to 'base'.
    :type sam_choice: str, optional
    """
    def __init__(self, sam_path: str, sam_choice: str = 'base'):
        """Initializes the PlaySegmentCallback.

        :param sam_path: Path to SAM2 model directory.
        :param sam_choice: SAM2 model variant to load.
        """
        super().__init__()
        self.sam_path = sam_path
        self._clear()
        self.sam_choice = sam_choice # Corrected assignment
        self._load_sam()

        # TODO: add different segment types

    def _load_sam(self):
        """Loads the specified SAM2 model checkpoint and configuration.

        Dynamically imports `build_sam2_camera_predictor` from `sam2.build_sam`
        and initializes the predictor.
        """
        ckpt_mapping = {
            'large': [os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"],
            'base': [os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"],
            'small': [os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"], 
            'tiny': [os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"]
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        # first realease the old predictor
        if hasattr(self, "predictor"):
            del self.predictor
        from sam2.build_sam import build_sam2_camera_predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, sam_ckpt)
        print(f"Successfully loaded SAM2 from {sam_ckpt}")
        self.able_to_track = False

    def _get_message(self, info: Dict) -> Dict:
        """Constructs a message string for GUI display about segmentation status.

        :param info: The current info dictionary.
        :type info: Dict
        :returns: The modified info dictionary with the segmentation message.
        :rtype: Dict
        """
        message = info.get('message', {})
        message['SegmentCallback'] = f'Segment: {"On" if self.tracking else "Off"}, Tracking Time: {self.tracking_time}'
        return message

    def _clear(self):
        """Resets all segmentation-related state variables."""
        self.positive_points = []
        self.negative_points = []
        self.segment = None
        self.able_to_track = False
        self.tracking = False
        self.tracking_time = 0

    def after_reset(self, sim, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
        """Clears segmentation state and adds a GUI message after environment reset.

        :param sim: The simulator instance.
        :param obs: The initial observation.
        :param info: The initial info dictionary.
        :returns: The passed `obs` and `info`.
        :rtype: Tuple[Dict, Dict]
        """
        self._clear()
        sim.callback_messages.add("Press 'S' to start/stop segmenting.")
        info['message'] = self._get_message(info)
        return obs, info

    def before_step(self, sim, action: Any) -> Any:
        """Prevents actions if segmentation GUI is active but not yet tracking.

        If the 'S' key was pressed to start segmentation but tracking hasn't begun
        (i.e., user is still providing prompts), this returns a no-op action to
        pause game progression.

        :param sim: The simulator instance.
        :param action: The proposed action.
        :type action: Any
        :returns: A no-op action if segmenting GUI is active, else the original action.
        :rtype: Any
        """
        if hasattr(sim, 'info') and sim.info.get('S', False) and not self.tracking:
            return sim.noop_action()
        return action
    
    def after_step(self, sim, obs: Dict, reward: float, terminated: bool, truncated: bool, info: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Manages the segmentation lifecycle based on user input ('S' key) and GUI interaction.

        Handles:
        - Starting segmentation GUI when 'S' is pressed and not already tracking.
        - Stopping tracking when 'S' is pressed while tracking.
        - Updating the segmentation mask in `obs` if tracking is active.

        :param sim: The simulator instance.
        :param obs: The observation after the step.
        :param reward: The reward received.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: The info dictionary.
        :returns: The (potentially modified) obs, reward, terminated, truncated, and info.
        :rtype: Tuple[Dict, float, bool, bool, Dict]
        """
        if self.tracking and (not info.get('S', False)):
            # stop tracking
            print(f'[red]Stop tracking[/red]')
            self._clear()
            info['segment'] = None
        elif (not self.tracking) and info.get('S', False):
            # start tracking
            print(f'[green]Start segmenting[/green]')
            current_info = info.copy() # Use a copy for the GUI
            current_info['segment'] = None
            current_info['positive_points'] = []
            current_info['negative_points'] = []
            current_info = self._segment_gui(current_info, sim) # Pass sim to _segment_gui
            # Update original info based on GUI results if necessary
            info['segment'] = current_info.get('segment')
            info['positive_points'] = current_info.get('positive_points', [])
            info['negative_points'] = current_info.get('negative_points', [])
            if not self.tracking:
                info['S'] = False
        elif self.tracking and info.get('S', False):
            self.tracking_time += 1
            info['segment'] = self._segment(info)

        if info.get('segment', None) is not None and self.tracking:
            # resize the segment to the size of the obs
            segment = cv2.resize(info['segment'].astype(np.uint8), dsize=(obs['image'].shape[0], obs['image'].shape[1]), interpolation=cv2.INTER_NEAREST)
            obs['segment'] = {}
            obs['segment']['obj_mask'] = segment
            obs['segment']['obj_id'] = 2
        else:
            obs['segment'] = {}
            obs['segment']['obj_mask'] = np.zeros((obs['image'].shape[0], obs['image'].shape[1]), dtype=np.uint8)
            obs['segment']['obj_id'] = -1
        
        info['message'] = self._get_message(info)
        return obs, reward, terminated, truncated, info
        
    def _segment_gui(self, current_info: Dict, sim) -> Dict:
        """Manages the GUI for interactive point-based segmentation.

        This method creates a GUI window where the user can add positive and
        negative points on the current POV. It updates the segmentation mask
        in real-time based on these prompts.

        Controls:
        - Left Click: Add positive point.
        - Right Click: Add negative point.
        - 'C': Clear points.
        - 'Enter': Finalize points and start tracking.
        - 'ESCAPE': Cancel and exit segmentation GUI.

        :param current_info: A copy of the current info dictionary, used for GUI display.
        :type current_info: Dict
        :param sim: The simulator instance (passed to access POV for segmentation).
        :type sim: Any
        :returns: The `current_info` dictionary, potentially updated with segmentation results.
        :rtype: Dict
        """
        info = current_info.copy()
        gui = MinecraftGUI(extra_draw_call=[SegmentDrawCall, MultiPointDrawCall], show_info=True)
        help_message = [["Press 'C' to clear points."], ["Press mouse left button to add points."], ["Press mouse right button to add negative points."], ["Press 'Enter' to start tracking."], ["Press 'ESC' to exit."]]

        gui.window.activate()
        refresh = False
        last_mouse_position = None

        while True:
            gui.window.dispatch_events()
            gui.window.switch_to()
            gui.window.set_mouse_visible(True)
            gui.window.set_exclusive_mouse(False)
            gui.window.flip()

            released_keys = gui._capture_all_keys()
            if 'ESCAPE' in released_keys:
                self._clear()
                info['segment'] = None
                info['positive_points'] = self.positive_points
                info['negative_points'] = self.negative_points
                self.tracking = False
                print('[red]Exit segmenting[/red]')
                break

            if 'C' in released_keys:
                self._clear()
                info['segment'] = None
                info['positive_points'] = self.positive_points
                info['negative_points'] = self.negative_points
                last_mouse_position = None
                refresh = True
                print('[red]Points cleared[/red]')

            if 'ENTER' in released_keys and self.able_to_track:
                assert info['segment'] is not None, 'segment is not generated before tracking.'
                print(f'[green]Start tracking[/green]')
                self.tracking = True
                break

            if gui.mouse_position is not None:
                if gui.mouse_pressed == 1 or gui.mouse_pressed == 4:
                    if gui.mouse_position != last_mouse_position:
                        last_mouse_position = gui.mouse_position
                        # Adjust for info bar height if GUI shows it
                        y_offset = gui.constants.INFO_HEIGHT if gui.show_info else 0
                        position = (last_mouse_position[0], gui.constants.FRAME_HEIGHT + y_offset - last_mouse_position[1])
                        
                        # Ensure pov_shape is derived correctly from sim.obs or info
                        # Assuming info contains the pov under 'pov' key as per _segment method context
                        pov_image = info.get('pov', sim.obs.get('pov'))
                        if pov_image is None:
                            print("[red]POV image not found in info or sim.obs for segmentation GUI.[/red]")
                            # Handle error or return, as POV is crucial
                            gui.close_gui()
                            return info
                        pov_shape = pov_image.shape

                        position = (int(position[0] * pov_shape[1] / gui.constants.WINDOW_WIDTH),
                                    int(position[1] * pov_shape[0] / gui.constants.FRAME_HEIGHT))

                        if gui.mouse_pressed == 1:
                            # left button pressed
                            self.positive_points.append(position)
                            info['positive_points'] = self.positive_points
                            print(f'[green]Positive point added at {position}[/green]')
                            refresh = True
                        elif gui.mouse_pressed == 4:
                            # right button pressed
                            self.negative_points.append(position)
                            info['negative_points'] = self.negative_points
                            print(f'[red]Negative point added at {position}[/red]')
                            refresh = True
                    gui.mouse_pressed = 0

            if len(self.positive_points) > 0:
                self.able_to_track = True

            if self.able_to_track:
                self._segment(info, refresh)
                info['segment'] = self.segment
                refresh = False

            gui._update_image(info, message=help_message, remap_points=(gui.constants.WINDOW_WIDTH, pov_shape[1], gui.constants.FRAME_HEIGHT, pov_shape[0]))

        gui.close_gui()
        return info

    def _segment(self, current_info: Dict, refresh: bool = False):
        """Performs segmentation using the SAM2 predictor.

        If `self.segment` is None or `refresh` is True, it loads the first frame
        and adds new prompts. Otherwise, it tracks the existing segment on the new POV.

        :param current_info: The dictionary containing the POV image ('pov') and points.
        :type current_info: Dict
        :param refresh: Whether to re-initialize segmentation with current points.
                        Defaults to False.
        :type refresh: bool, optional
        :returns: The generated segmentation mask.
        :rtype: np.ndarray
        """
        pov_image = current_info.get('pov')
        if pov_image is None:
            # Attempt to get POV from sim.obs if not in current_info (e.g. during tracking)
            # This part depends on how sim object is available or if obs is passed differently
            # For now, let's assume it must be in current_info for _segment
            print("[red]POV image not found in current_info for _segment.[/red]")
            return self.segment # Return existing segment or None

        if (self.segment is None) or refresh:
            assert len(self.positive_points) > 0
            points = self.positive_points + self.negative_points
            self.predictor.load_first_frame(pov_image)
            _, out_obj_ids, out_segment_logits = self.predictor.add_new_prompt(
                frame_idx=0, 
                obj_id=0,
                points=points,
                labels=[1] * len(self.positive_points) + [0] * len(self.negative_points),
            )
        else:
            out_obj_ids, out_segment_logits = self.predictor.track(pov_image)
        self.segment = (out_segment_logits[0, 0] > 0.0).cpu().numpy() # 360 * 640
        return self.segment









