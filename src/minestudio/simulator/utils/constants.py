'''
Date: 2024-11-15 14:57:39
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-17 22:32:08
FilePath: /Minestudio/minestudio/simulator/utils/constants.py
'''

class GUIConstants:
    """
    Defines constants used for the Minecraft GUI.

    This includes keyboard and mouse mappings for actions, display dimensions,
    FPS settings, and grid layout parameters for the information panel.
    """
    def __init__(self):
        """
        Initializes GUIConstants.

        Imports Pyglet and sets up action mappings, mouse sensitivity,
        FPS, window dimensions (scaled to screen size if necessary),
        and info panel grid layout.
        """
        import pyglet
        import pyglet.window.key as key
        import pyglet.window.mouse as mouse

        # Mapping from simulator's action space names to pyglet keys
        self.MINERL_ACTION_TO_KEYBOARD = {
            #"ESC":       key.ESCAPE, # Used in BASALT to end the episode
            "attack":    mouse.LEFT,
            "back":      key.S,
            #"drop":      key.Q,
            "forward":   key.W,
            "hotbar.1":  key._1,
            "hotbar.2":  key._2,
            "hotbar.3":  key._3,
            "hotbar.4":  key._4,
            "hotbar.5":  key._5,
            "hotbar.6":  key._6,
            "hotbar.7":  key._7,
            "hotbar.8":  key._8,
            "hotbar.9":  key._9,
            "inventory": key.E,
            "jump":      key.SPACE,
            "left":      key.A,
            # "pickItem":  pyglet.window.mouse.MIDDLE,
            "right":     key.D,
            "sneak":     key.LSHIFT,
            "sprint":    key.LCTRL,
            #"swapHands": key.F,
            "use":       mouse.RIGHT, 
            # "switch":    key.TAB,
            # "reset":     key.F1,
        }

        self.KEYBOARD_TO_MINERL_ACTION = {v: k for k, v in self.MINERL_ACTION_TO_KEYBOARD.items()}

        self.IGNORED_ACTIONS = {"chat"}

        # Camera actions are in degrees, while mouse movement is in pixels
        # Multiply mouse speed by some arbitrary multiplier
        self.MOUSE_MULTIPLIER = 0.1

        self.MINERL_FPS = 25
        self.MINERL_FRAME_TIME = 1 / self.MINERL_FPS

        self.SCALE = 2
        self.WINDOW_WIDTH = 640 * self.SCALE
        self.WINDOW_HEIGHT = 360 * (self.SCALE + 1)

        screen = pyglet.canvas.get_display().get_default_screen()
        ratio = 0.8

        if screen.width < self.WINDOW_WIDTH * ratio or screen.height < self.WINDOW_HEIGHT * ratio:
            scale = min(screen.width * ratio / self.WINDOW_WIDTH, screen.height * ratio / self.WINDOW_HEIGHT)
            self.WINDOW_WIDTH = int(self.WINDOW_WIDTH * scale)
            self.WINDOW_HEIGHT = int(self.WINDOW_HEIGHT * scale)

        self.FRAME_HEIGHT = self.WINDOW_HEIGHT // 4 * 3

        self.INFO_WIDTH = self.WINDOW_WIDTH
        self.INFO_HEIGHT = self.WINDOW_HEIGHT // 4

        self.NUM_ROWS = 4
        self.NUM_COLS = 6
        self.GRID_WIDTH = self.WINDOW_WIDTH // self.NUM_COLS
        self.GRID_HEIGHT = self.INFO_HEIGHT // self.NUM_ROWS
        self.GRID = {}
        self.GRID_ID = 0
        for R in range(self.NUM_ROWS):
            for C in range(self.NUM_COLS):
                X = C * self.GRID_WIDTH + self.GRID_WIDTH // 5
                Y = R * self.GRID_HEIGHT + self.GRID_HEIGHT // 2
                self.GRID[self.GRID_ID] = (X, Y)
                self.GRID_ID += 1
