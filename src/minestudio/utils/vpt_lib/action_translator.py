import numpy as np

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}
 
# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}
 
MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.
 
Press any button the window to proceed to the next frame.
"""
 
# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0
 
 
 
class CameraQuantizer():
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.
    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """
    def __init__(self, camera_maxval = 10, camera_binsize = 2, quantization_scheme = "mu_law", mu = 10):
        self.camera_maxval = camera_maxval
        self.camera_binsize = camera_binsize
        self.quantization_scheme = quantization_scheme
        self.mu = mu
 
    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)
 
        if self.quantization_scheme == "mu_law":
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode
 
        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)
 
    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval
 
        if self.quantization_scheme == "mu_law":
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy
 
 
def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
 
    Note: in some recordings, the recording starts with "attack: 1" despite
    player obviously not attacking. The IDM seems to reflect this (predicts "attack: 1")
    at the beginning of the recording.
    """
    env_action = NOOP_ACTION.copy()
    env_action["camera"] = np.array([0, 0])
 
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
 
    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER
 
    if abs(camera_action[0]) > 180:
        camera_action[0] = 0
    if abs(camera_action[1]) > 180:
        camera_action[1] = 0
 
    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
    if 1 in mouse_buttons:
        env_action["use"] = 1
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
 
    return env_action
 
 
def translate_action_to_dojo(action):
 
    noop = np.array([0, 0, 0, 5, 5, 0, 0, 0]).astype("int64")
    if action.get('forward', 0) == 1:
        noop[0] = 1
    if action.get('back', 0) == 1:
        noop[0] = 2
 
    if action.get('left', 0) == 1:
        noop[1] = 1
    if action.get('right', 0) == 1:
        noop[1] = 2
 
    if action.get('jump', 0) == 1:
        noop[2] = 1
 
    if action.get('sneak', 0) == 1:
        noop[2] = 2
 
    if action.get('sprint', 0) == 1:
        noop[2] = 3
 
    if action.get('attack', 0) == 1:
        noop[5] = 3
 
    if action.get('use', 0) == 1:
        noop[5] = 1
 
    # if action.get('drop', 0) == 1:
    #     noop[5] = 7
 
    camera = action.get('camera', np.array([0., 0.]))
    if isinstance(camera[0], float):
        noop[3] = CameraQuantizer().discretize(camera[0])
        noop[4] = CameraQuantizer().discretize(camera[1])
    else:
        noop[3] = CameraQuantizer().discretize(camera[0][0])
        noop[4] = CameraQuantizer().discretize(camera[0][1])
    return noop