'''
Date: 2024-11-15 15:15:22
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-20 01:02:18
FilePath: /MineStudio/minestudio/simulator/utils/gui.py
'''
from minestudio.simulator.utils.constants import GUIConstants   

from collections import defaultdict
from typing import List, Any, Optional, Callable
import importlib
import cv2
import time
from rich import print
import numpy as np

def RecordDrawCall(info, **kwargs):
    """
    Draws a recording indicator on the POV display.

    A red or green circle and "Rec" text are drawn on the top-left corner
    of the POV image if recording is active. The color of the circle
    alternates based on the current time.

    :param info: A dictionary containing the 'pov' image and 'R' (recording status) flag.
    :param kwargs: Additional keyword arguments (not used).
    :return: The modified info dictionary with the recording indicator drawn on the 'pov' image.
    """
    if 'R' not in info.keys() or info.get('ESCAPE', False):
        return info
    recording = info['R']
    if not recording:
        return info
    arr = info['pov']
    if int(time.time()) % 2 == 0:
        cv2.circle(arr, (20, 20), 10, (255, 0, 0), -1)
        cv2.putText(arr, 'Rec', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.circle(arr, (20, 20), 10, (0, 255, 0), -1)
        cv2.putText(arr, 'Rec', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    info['pov'] = arr
    return info

def CommandModeDrawCall(info, **kwargs):
    """
    Draws a command mode indicator on the POV display.

    If command mode is active (indicated by the 'ESCAPE' flag in info),
    the POV image is converted to grayscale, and "Command Mode" text is
    drawn on the top-left corner.

    :param info: A dictionary containing the 'pov' image and 'ESCAPE' (command mode) flag.
    :param kwargs: Additional keyword arguments (not used).
    :return: The modified info dictionary with the command mode indicator.
    """
    if 'ESCAPE' not in info.keys():
        return info
    mode = info['ESCAPE']
    if not mode:
        return info
    # Draw a grey overlay on the screen
    arr = info['pov']
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    cv2.putText(arr, 'Command Mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    info['pov'] = arr
    return info

def PointDrawCall(info, **kwargs):
    """
    Draws a point indicator on the POV display.

    If a 'point' is present in the info dictionary, a red circle is drawn
    at the specified coordinates on the POV image. Text indicating the
    point's coordinates is also displayed.

    :param info: A dictionary containing the 'pov' image and 'point' coordinates.
    :param kwargs: Additional keyword arguments (not used).
    :return: The modified info dictionary with the point drawn on the 'pov' image.
    """
    if 'point' not in info.keys():
        return info
    point = info['point']
    arr  = info['pov']
    # draw a red circle at the point, the position is relative to the bottom-left corner of arr
    cv2.circle(arr, (point[0], arr.shape[0] - point[1]), 10, (0, 0, 255), -1)
    cv2.putText(arr, f'Pointing at {point}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    info['pov'] = arr
    return info

def MultiPointDrawCall(info, **kwargs):
    """
    Draws multiple point indicators (positive and negative) on the POV display.

    Positive points are drawn as green circles, and negative points are
    drawn as red circles. Point coordinates can be remapped using 'remap_points'
    in kwargs.

    :param info: A dictionary containing 'positive_points' and 'negative_points' lists,
                 and the 'pov' image.
    :param kwargs: Additional keyword arguments, including 'remap_points'.
    :return: The modified info dictionary with points drawn on the 'pov' image.
    """
    if 'positive_points' not in info.keys() or 'negative_points' not in info.keys():
        return info
    positive_points = info['positive_points']
    negative_points = info['negative_points']
    if len(positive_points) == 0:
        return info
    arr = info['pov']
    remap_points = kwargs.get('remap_points', (1, 1, 1, 1))
    for point in positive_points:
        point = (int(point[0] * remap_points[0] / remap_points[1]), int(point[1] * remap_points[2] / remap_points[3]))
        cv2.circle(arr, (point[0], point[1]), 10, (0, 255, 0), -1)

    for point in negative_points:
        point = (int(point[0] * remap_points[0] / remap_points[1]), int(point[1] * remap_points[2] / remap_points[3]))
        cv2.circle(arr, (point[0], point[1]), 10, (255, 0, 0), -1)

    info['pov'] = arr
    return info

def SegmentDrawCall(info, **kwargs):
    """
    Draws a segmentation mask overlay on the POV display.

    If a 'segment' mask is present in the info dictionary, it's resized
    to the POV image dimensions and overlaid with a green color.

    :param info: A dictionary containing the 'segment' mask and 'pov' image.
    :param kwargs: Additional keyword arguments (not used).
    :return: The modified info dictionary with the segmentation mask overlay.
    """
    if 'segment' not in info.keys():
        return info
    mask = info['segment']
    if mask is None:
        return info
    arr = info['pov']
    color = (0, 255, 0)
    color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
    mask = (mask[..., None] * color).astype(np.uint8)
    # resize the mask to the size of the obs
    mask = cv2.resize(mask, dsize=(arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
    arr = cv2.addWeighted(arr, 1.0, mask, 0.5, 0.0)
    info['pov'] = arr
    return info
    
class MinecraftGUI:
    """
    Manages the Pyglet-based GUI for the Minecraft simulator.

    Handles window creation, event processing (keyboard, mouse),
    rendering of the POV display, and displaying informational messages.
    It also supports custom draw calls for additional visual elements.
    """
    def __init__(self, extra_draw_call: List[Callable] = None, show_info = True, **kwargs):
        """
        Initializes the MinecraftGUI.

        :param extra_draw_call: A list of callable functions for custom drawing on the POV.
        :param show_info: Boolean indicating whether to display the information panel.
        :param kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(**kwargs)
        self.constants = GUIConstants()
        self.pyglet = importlib.import_module('pyglet')
        self.imgui = importlib.import_module('imgui')
        self.key = importlib.import_module('pyglet.window.key')
        self.mouse = importlib.import_module('pyglet.window.mouse')
        self.PygletRenderer = importlib.import_module('imgui.integrations.pyglet').PygletRenderer
        self.extra_draw_call = extra_draw_call
        self.show_info = show_info
        self.mode = 'normal'
        self.create_window()
    
    def create_window(self):
        """
        Creates the Pyglet window and sets up event handlers and ImGui integration.
        """
        if self.show_info:
            self.window = self.pyglet.window.Window(
                width = self.constants.WINDOW_WIDTH,
                height = self.constants.INFO_HEIGHT + self.constants.FRAME_HEIGHT,
                vsync=False,
                resizable=False
            )
        else:
            self.window = self.pyglet.window.Window(
                width = self.constants.WINDOW_WIDTH,
                height = self.constants.FRAME_HEIGHT,
                vsync=False,
                resizable=False
            )
        self.imgui.create_context()
        self.imgui.get_io().display_size = self.constants.WINDOW_WIDTH, self.constants.WINDOW_HEIGHT
        self.renderer = self.PygletRenderer(self.window)
        self.pressed_keys = defaultdict(lambda: False)
        self.released_keys = defaultdict(lambda: False)
        self.modifiers = None
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactivate = self._on_window_deactivate

        self.last_pov = None
        self.last_mouse_delta = [0, 0]
        self.capture_mouse = True
        self.mouse_position = None
        self.mouse_pressed = None
        self.chat_message = None
        self.command = None

        self.window.dispatch_events()
        self.window.switch_to()
        self.window.flip()
        self.window.clear()

        self._show_message("Waiting for start.")

    def _on_key_press(self, symbol, modifiers):
        """
        Handles key press events.

        :param symbol: The Pyglet key symbol.
        :param modifiers: Key modifiers (e.g., Shift, Ctrl).
        """
        self.pressed_keys[symbol] = True
        self.modifiers = modifiers

    def _on_key_release(self, symbol, modifiers):
        """
        Handles key release events.

        :param symbol: The Pyglet key symbol.
        :param modifiers: Key modifiers.
        """
        self.pressed_keys[symbol] = False
        self.released_keys[symbol] = True
        self.modifiers = modifiers

    def _on_mouse_press(self, x, y, button, modifiers):
        """
        Handles mouse button press events.

        :param x: The x-coordinate of the mouse press.
        :param y: The y-coordinate of the mouse press.
        :param button: The mouse button pressed.
        :param modifiers: Key modifiers.
        """
        self.pressed_keys[button] = True
        self.mouse_pressed = button
        self.mouse_position = (x, y)

    def _on_mouse_release(self, x, y, button, modifiers):
        """
        Handles mouse button release events.

        :param x: The x-coordinate of the mouse release.
        :param y: The y-coordinate of the mouse release.
        :param button: The mouse button released.
        :param modifiers: Key modifiers.
        """
        self.pressed_keys[button] = False

    def _on_window_activate(self):
        """
        Handles window activation events (e.g., window gains focus).

        Sets mouse visibility and exclusivity for gameplay.
        """
        self.window.set_mouse_visible(False)
        self.window.set_exclusive_mouse(True)

    def _on_window_deactivate(self):
        """
        Handles window deactivation events (e.g., window loses focus).

        Restores mouse visibility and exclusivity.
        """
        self.window.set_mouse_visible(True)
        self.window.set_exclusive_mouse(False)

    def _on_mouse_motion(self, x, y, dx, dy):
        """
        Handles mouse motion events.

        Updates the `last_mouse_delta` for camera control. Note that
        vertical mouse movement (dy) is inverted.

        :param x: The current x-coordinate of the mouse.
        :param y: The current y-coordinate of the mouse.
        :param dx: The change in x-coordinate since the last event.
        :param dy: The change in y-coordinate since the last event.
        """
        # Inverted
        self.last_mouse_delta[0] -= dy * self.constants.MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * self.constants.MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Handles mouse drag events (mouse motion while a button is pressed).

        Updates the `last_mouse_delta` for camera control. Note that
        vertical mouse movement (dy) is inverted.

        :param x: The current x-coordinate of the mouse.
        :param y: The current y-coordinate of the mouse.
        :param dx: The change in x-coordinate since the last event.
        :param dy: The change in y-coordinate since the last event.
        :param buttons: The mouse buttons currently pressed.
        :param modifiers: Key modifiers.
        """
        # Inverted
        self.last_mouse_delta[0] -= dy * self.constants.MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * self.constants.MOUSE_MULTIPLIER

    def _show_message(self, text):
        """
        Displays a centered message on the screen.

        Used for messages like "Waiting for start." or "Resetting environment...".

        :param text: The text to display.
        """
        document = self.pyglet.text.document.FormattedDocument(text)
        document.set_style(0, len(document.text), dict(font_name='Arial', font_size=32, color=(255, 255, 255, 255)))
        document.set_paragraph_style(0,100,dict(align = 'center'))
        layout = self.pyglet.text.layout.TextLayout(
            document,
            width=self.window.width//2,
            height=self.window.height//2,
            multiline=True,
            wrap_lines=True,
        )
        layout.update(x=self.window.width//2, y=self.window.height//2)
        layout.anchor_x = 'center'
        layout.anchor_y = 'center'
        layout.content_valign = 'center'
        layout.draw()

        self.window.flip()

    def _show_additional_message(self, message: List):
        """
        Displays additional messages in the info panel.

        Each item in the `message` list is displayed as a row of text.

        :param message: A list of lists/tuples, where each inner list/tuple
                        represents a row of text elements to be joined by " | ".
        """
        if len(message) == 0:
            return
        line_height = self.constants.INFO_HEIGHT // len(message)
        y = line_height // 2
        for i, row in enumerate(message):
            line = ' | '.join(row)
            self.pyglet.text.Label(
                line,
                font_size = 7 * self.constants.SCALE, 
                x = self.window.width // 2, y = y, 
                anchor_x = 'center', anchor_y = 'center',
            ).draw()
            y += line_height

    def _update_image(self, info, message: List = [], **kwargs):
        """
        Updates and renders the main game view (POV) and info panel.

        Resizes the POV, applies extra draw calls, displays additional messages,
        and handles ImGui rendering for the chat interface.

        :param info: A dictionary containing the 'pov' image.
        :param message: A list of messages for the info panel.
        :param kwargs: Additional keyword arguments for extra draw calls.
        """
        self.window.switch_to()
        self.window.clear()
        # Based on scaled_image_display.py
        info = info.copy()
        arr = info['pov']
        arr = cv2.resize(arr, dsize=(self.constants.WINDOW_WIDTH, self.constants.FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC) # type: ignore
        info['pov'] = arr
        
        if self.extra_draw_call is not None:
            for draw_call in self.extra_draw_call:
                info = draw_call(info, **kwargs)

        arr = info['pov']
        image = self.pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, self.constants.INFO_HEIGHT)

        if self.show_info:
            self._show_additional_message(message)
        
        self.imgui.new_frame()
        
        self.imgui.begin("Chat", False, self.imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        changed, command = self.imgui.input_text("Message", "")
        self.command = command
        if self.imgui.button("Send"):
            self.chat_message = command
            self.command = None
        self.imgui.end()

        self.imgui.render()
        self.renderer.render(self.imgui.get_draw_data())
        self.window.flip()

    def _show_image(self, info, **kwargs):
        """
        Displays the POV image without the info panel or ImGui chat.

        Used when `show_info` is False.

        :param info: A dictionary containing the 'pov' image.
        :param kwargs: Additional keyword arguments for extra draw calls.
        """
        self.window.switch_to()
        self.window.clear()
        info = info.copy()
        arr = info['pov']
        arr = cv2.resize(arr, dsize=(self.constants.WINDOW_WIDTH, self.constants.FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
        info['pov'] = arr
        if self.extra_draw_call is not None:
            for draw_call in self.extra_draw_call:
                info = draw_call(info, **kwargs)
        arr = info['pov']
        image = self.pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, 0)
        self.window.flip()

    def _get_human_action(self):
        """
        Reads keyboard and mouse state to form a human action dictionary.

        :return: A dictionary representing the current human action.
        """
        # Keyboard actions
        action: dict[str, Any] = {
            name: int(self.pressed_keys[key]) for name, key in self.constants.MINERL_ACTION_TO_KEYBOARD.items()
        }

        if not self.capture_mouse:
            self.last_mouse_delta = [0, 0]
        action["camera"] = self.last_mouse_delta
        self.last_mouse_delta = [0, 0]
        return action
        
    def reset_gui(self):
        """
        Resets the GUI state, clears the window, and shows a "Resetting" message.
        """
        self.window.clear()
        self.pressed_keys = defaultdict(lambda: False)
        self._show_message("Resetting environment...")

    def _capture_all_keys(self):
        """
        Captures all keys that were released since the last call.

        :return: A set of string representations of the released keys.
        """
        released_keys = set()
        for key in self.released_keys.keys():
            if self.released_keys[key]:
                self.released_keys[key] = False
                released_keys.add(self.key.symbol_string(key))
        return released_keys

    def close_gui(self):
        """
        Closes the Pyglet window and exits the Pyglet application.
        """
        #! WARNING: This should be checked
        self.window.close()
        self.pyglet.app.exit()

if __name__ == "__main__":
    gui = MinecraftGUI()