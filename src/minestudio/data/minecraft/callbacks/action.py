'''
Date: 2025-01-09 05:27:25
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-27 15:04:29
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/action.py
'''
import re
import cv2
import pickle
import numpy as np
from pathlib import Path
from rich import print
from tqdm import tqdm
from typing import Union, Tuple, List, Dict, Callable, Any, Literal
from collections import OrderedDict

from minestudio.utils.vpt_lib.actions import ActionTransformer
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, DrawFrameCallback, ModalConvertCallback
from minestudio.utils.register import Registers

@Registers.modal_kernel_callback.register
class ActionKernelCallback(ModalKernelCallback):
    """
    Callback for handling Minecraft actions.

    This callback processes action data, including decoding, merging, slicing, padding, and postprocessing.
    It can handle both regular actions and previous actions if enabled.
    """

    def create_from_config(config: Dict) -> 'ActionKernelCallback':
        """
        Creates an ActionKernelCallback instance from a configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict
        :returns: An instance of ActionKernelCallback.
        :rtype: ActionKernelCallback
        """
        return ActionKernelCallback(**config.get('action', {}))

    def __init__(self, 
                 n_camera_bins: int=11,
                 camera_binsize: int=2,
                 camera_maxval: int=10,
                 camera_mu: int=10,
                 camera_quantization_scheme="mu_law", 
                 enable_prev_action: bool=False,
                 **kwargs):
        """
        Initializes the ActionKernelCallback.

        :param n_camera_bins: Number of bins for camera quantization.
        :type n_camera_bins: int
        :param camera_binsize: Binsize for camera quantization.
        :type camera_binsize: int
        :param camera_maxval: Maximum value for camera quantization.
        :type camera_maxval: int
        :param camera_mu: Mu value for camera mu-law quantization.
        :type camera_mu: int
        :param camera_quantization_scheme: Scheme for camera quantization ("mu_law" or other).
        :type camera_quantization_scheme: str
        :param enable_prev_action: Whether to enable processing of previous actions.
        :type enable_prev_action: bool
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=n_camera_bins)
        self.action_transformer = ActionTransformer(
            camera_binsize=camera_binsize,
            camera_maxval=camera_maxval,
            camera_mu=camera_mu,
            camera_quantization_scheme=camera_quantization_scheme
        )
        self.enable_prev_action = enable_prev_action

    @property
    def name(self) -> str:
        """
        Returns the name of the callback.

        :returns: The name 'action'.
        :rtype: str
        """
        return 'action'

    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        """
        Filters dataset paths to select only action-related paths.

        :param dataset_paths: A list of dataset paths.
        :type dataset_paths: List[Union[str, Path]]
        :returns: A list of paths pointing to action data.
        :rtype: List[Path]
        """
        if isinstance(dataset_paths[0], str):
            dataset_paths = [Path(path) for path in dataset_paths]
        action_paths = [path for path in dataset_paths if Path(path).stem == 'action']
        return action_paths

    def do_decode(self, chunk: bytes, **kwargs) -> Dict:
        """
        Decodes a chunk of bytes into an action dictionary.

        :param chunk: Bytes to decode.
        :type chunk: bytes
        :param kwargs: Additional keyword arguments.
        :returns: Decoded action dictionary.
        :rtype: Dict
        """
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Dict:
        """
        Merges a list of decoded action chunks into a single dictionary.

        :param chunk_list: List of byte chunks representing actions.
        :type chunk_list: List[bytes]
        :param kwargs: Additional keyword arguments.
        :returns: A dictionary containing merged action data.
        :rtype: Dict
        """
        chunks = [self.do_decode(chunk) for chunk in chunk_list]
        cache_chunks = {}
        for chunk in chunks:
            for key, value in chunk.items():
                if key not in cache_chunks:
                    cache_chunks[key] = []
                cache_chunks[key].append(value)
        merged_chunks = {key: np.concatenate(value, axis=0) for key, value in cache_chunks.items()}
        return merged_chunks

    def do_slice(self, data: Dict, start: int, end: int, skip_frame: int, **kwargs) -> Dict:
        """
        Slices the action data.

        :param data: Action data dictionary.
        :type data: Dict
        :param start: Start index for slicing.
        :type start: int
        :param end: End index for slicing.
        :type end: int
        :param skip_frame: Frame skipping interval.
        :type skip_frame: int
        :param kwargs: Additional keyword arguments.
        :returns: Sliced action data.
        :rtype: Dict
        """
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Dict, np.ndarray]:
        """
        Pads the action data.

        :param data: Action data dictionary.
        :type data: Dict
        :param pad_len: Length of padding to add.
        :type pad_len: int
        :param pad_pos: Position to add padding ("left" or "right").
        :type pad_pos: Literal["left", "right"]
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing the padded action data and the padding mask.
        :rtype: Tuple[Dict, np.ndarray]
        """
        pad_data = dict()
        for key, value in data.items():
            traj_len = value.shape[0]
            dims = value.shape[1:]
            if pad_pos == "right":
                pad_value = np.concatenate([value, np.zeros((pad_len, *dims), dtype=np.uint8)], axis=0)
            elif pad_pos == "left":
                pad_value = np.concatenate([np.zeros((pad_len, *dims), dtype=np.uint8), value], axis=0)
            pad_data[key] = pad_value
        if pad_pos == "right":
            pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(pad_len, dtype=np.uint8)], axis=0)
        else:
            pad_mask = np.concatenate([np.zeros(pad_len, dtype=np.uint8), np.ones(traj_len, dtype=np.uint8)], axis=0)
        return pad_data, pad_mask

    def do_postprocess(self, data: Dict) -> Dict:
        """
        Postprocesses the action data.

        This method handles the transformation of environment actions to agent actions
        and optionally includes previous actions.

        :param data: Data dictionary containing actions.
        :type data: Dict
        :returns: Postprocessed data dictionary.
        :rtype: Dict
        """
        pop_action = data.pop(self.name)
        if not self.enable_prev_action:
            data[f'env_{self.name}'] = pop_action
            data[f'agent_{self.name}'] = self.action_mapper.from_factored(
                self.action_transformer.env2policy(pop_action)
            )
        else:
            action = { key: val[1:] for key, val in pop_action.items() }
            prev_action = { key: val[:-1] for key, val in pop_action.items() }
            data[f'env_{self.name}'] = action
            data[f'env_prev_{self.name}'] = prev_action
            data[f'agent_{self.name}'] = self.action_mapper.from_factored(
                self.action_transformer.env2policy(action)
            )
            data[f'agent_prev_{self.name}'] = self.action_mapper.from_factored(
                self.action_transformer.env2policy(prev_action)
            )
            data[f'{self.name}_mask'] = np.ones(len(action['attack']), dtype=np.uint8)
        return data

class VectorActionKernelCallback(ActionKernelCallback):
    """
    Callback for handling actions represented as vectors.

    This callback converts actions between dictionary and vector representations.
    """

    ACTION_KEYS = OrderedDict({
        'camera': 2, 'attack': 1, 'forward': 1, 'back': 1, 'left': 1, 'right': 1, 'jump': 1, 'sneak': 1, 'sprint': 1, 'use': 1, 'drop': 1, 'inventory': 1, 
        'hotbar.1': 1, 'hotbar.2': 1, 'hotbar.3': 1, 'hotbar.4': 1, 'hotbar.5': 1, 'hotbar.6': 1, 'hotbar.7': 1, 'hotbar.8': 1, 'hotbar.9': 1,
    })

    def __init__(self, action_chunk_size: int=32, return_type: str="vector"):
        """
        Initializes the VectorActionKernelCallback.

        :param action_chunk_size: Size of the action chunks.
        :type action_chunk_size: int
        :param return_type: The return type for processed actions ("vector" or "dict").
        :type return_type: str
        :raises AssertionError: if return_type is not "vector" or "dict".
        """
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.win_bias = action_chunk_size - 1
        assert return_type in ["vector", "dict"], f"Invalid return type: {return_type}"
        self.return_type = return_type

    @property
    def vector_dim(self) -> int:
        """
        Calculates the dimension of the action vector.

        :returns: The dimension of the action vector.
        :rtype: int
        """
        return sum(self.ACTION_KEYS.values()) * self.action_chunk_size

    def vector_to_action(self, vector: np.ndarray) -> Union[List[Dict], Dict]:
        """
        Converts an action vector or a list of action vectors to action dictionaries.

        :param vector: Action vector(s).
        :type vector: np.ndarray
        :returns: Action dictionary or list of action dictionaries.
        :rtype: Union[List[Dict], Dict]
        """
        if len(vector.shape) == 1:
            vector = vector[np.newaxis, ...]
        actions = []
        for i in range(vector.shape[0]):
            action = {key: [] for key in self.ACTION_KEYS}
            offset = 0
            for t in range(self.action_chunk_size):
                for idx, (key, dim) in enumerate(self.ACTION_KEYS.items()):
                    action[key].append(vector[i, offset: offset+dim])
                    offset += dim
            for key in action.keys():
                if key == 'camera':
                    action[key] = np.stack(action[key], axis=0) * 180.
                else:
                    action[key] = (np.concatenate(action[key], axis=0) >= 0).astype(np.uint8)
            actions.append(action)
        if len(vector.shape) == 1:
            return actions[0]
        return actions

    def action_to_vector(self, action: Dict) -> np.ndarray:
        """
        Converts an action dictionary to an action vector.

        :param action: Action dictionary.
        :type action: Dict
        :returns: Action vector.
        :rtype: np.ndarray
        """
        vectors = []
        win_len = len(action['attack'])
        for i in range(win_len - self.action_chunk_size + 1):
            vector = np.zeros(self.vector_dim, dtype=np.float32)
            offset = 0
            for t in range(self.action_chunk_size):
                for idx, (key, dim) in enumerate(self.ACTION_KEYS.items()):
                    if key == 'camera':
                        val = action[key][i+t] / 180
                    else:
                        val = action[key][i+t] * 2 - 1
                    vector[offset: offset+dim] = val
                    offset += dim
            vectors.append(vector)
        return np.stack(vectors, axis=0)

    def action_to_dict(self, action: Dict) -> Dict:
        """
        Converts an action dictionary to a specific dictionary format with "camera" and "button" keys.

        :param action: Action dictionary.
        :type action: Dict
        :returns: Dictionary with "camera" and "button" actions.
        :rtype: Dict
        """
        ret = {"camera": [], "button": []}
        win_len = len(action['attack'])
        for i in range(win_len - self.action_chunk_size + 1):
            camera, button = [], []
            for t in range(self.action_chunk_size):
                for key, dim in self.ACTION_KEYS.items():
                    if key == 'camera':
                        camera.extend(action[key][i+t] / 180)
                    else:
                        button.append(action[key][i+t] * 2 - 1)
            ret["camera"].append(camera)
            ret["button"].append(button)

        ret["camera"] = np.array(ret["camera"], dtype=np.float32)
        ret["button"] = np.array(ret["button"], dtype=np.float32)
        return ret

    def do_postprocess(self, data: Dict) -> Dict:
        """
        Postprocesses the action data, converting it to the specified return type (vector or dict).

        :param data: Data dictionary containing actions.
        :type data: Dict
        :returns: Postprocessed data dictionary with actions in the specified format.
        :rtype: Dict
        """
        action = data.pop('action')
        if self.return_type == "vector":
            ret = self.action_to_vector(action)
        elif self.return_type == "dict":
            ret = self.action_to_dict(action)
        # restored_action = self.vector_to_action(vector)
        # data['action_mask']: shape (128 + 32 -1, )
        # vector mask: shape (128, 32)
        action_chunk_mask = []
        for i in range(len(data['action_mask']) - self.action_chunk_size + 1):
            action_chunk_mask.append(data['action_mask'][i: i+self.action_chunk_size])
        action_chunk_mask = np.stack(action_chunk_mask, axis=0)
        data['action'] = ret
        data['action_chunk_mask'] = action_chunk_mask
        return data

class ActionDrawFrameCallback(DrawFrameCallback):
    """
    Callback for drawing action information onto video frames.
    """

    def __init__(self, start_point: Tuple[int, int]=(10, 10)):
        """
        Initializes the ActionDrawFrameCallback.

        :param start_point: The (x, y) coordinates for the top-left starting point of the text overlay.
        :type start_point: Tuple[int, int]
        """
        super().__init__()
        self.x, self.y = start_point

    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int) -> np.ndarray:
        """
        Draws action information onto each frame.

        :param frames: A list of frames or a numpy array of frames.
        :type frames: Union[np.ndarray, List]
        :param infos: Dictionary containing action information.
        :type infos: Dict
        :param sample_idx: Index of the sample to process.
        :type sample_idx: int
        :returns: A list of frames with action information drawn on them.
        :rtype: List[np.ndarray]
        """
        cache_frames = []
        env_action = infos['env_action']
        env_prev_action = infos.get('env_prev_action', env_action)
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            act = {k: v[sample_idx][frame_idx].numpy() for k, v in env_action.items()}
            prev_act = {k: v[sample_idx][frame_idx].numpy() for k, v in env_prev_action.items()}
            current_row = 0
            for (k, v), (_, pv) in zip(act.items(), prev_act.items()):
                if 'hotbar' in k:
                    continue
                if k != 'camera':
                    v = int(v.item())
                    pv = int(pv.item())
                else:
                    v = f"[{v[0].item():.3f}, {v[1].item():.3f}]"
                    pv = f"[{pv[0].item():.3f}, {pv[1].item():.3f}]"
                cv2.putText(frame, f"{k}: {v}({pv})", (self.x, self.y+35+current_row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                current_row += 1
            cache_frames.append(frame)
        return cache_frames


class ActionConvertCallback(ModalConvertCallback):
    """
    Callback for converting raw action data files.

    This callback loads action data from .jsonl files, processes it, and converts it into a structured format.
    """
    def __init__(self, 
                 input_dirs: List[Union[str, Path]], 
                 chunk_size: int=32, 
                 action_transformer_kwargs: Dict=dict(),
                 **kwargs):
        """
        Initializes the ActionConvertCallback.

        :param input_dirs: List of input directories containing action files.
        :type input_dirs: List[Union[str, Path]]
        :param chunk_size: Size of data chunks to process.
        :type chunk_size: int
        :param action_transformer_kwargs: Keyword arguments for the ActionTransformer.
        :type action_transformer_kwargs: Dict
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.input_dirs = input_dirs
        self.chunk_size = chunk_size
        self.action_transformer_kwargs = action_transformer_kwargs

    def load_episodes(self) -> OrderedDict:
        """
        Loads episodes from input directories containing .jsonl action files.

        It identifies episode segments from file names, sorts them, and organizes them into an OrderedDict.

        :returns: An OrderedDict where keys are episode IDs and values are lists of segment file paths.
        :rtype: OrderedDict
        """
        CONTRACTOR_PATTERN = r"^(.*?)-(\d+)$"
        episodes = OrderedDict()
        num_segments = 0
        for source_dir in self.input_dirs:
            print("Current input directory: ", source_dir) # action file ends with `.pkl`
            for file_path in tqdm(Path(source_dir).rglob("*.pkl"), desc="Looking for source files"):
                file_name = file_path.stem
                match = re.match(CONTRACTOR_PATTERN, file_name)
                if match:
                    eps, part_id = match.groups()
                else:
                    eps, part_id = file_name, "0"
                if eps not in episodes:
                    episodes[eps] = []
                episodes[eps].append( (part_id, file_path) )
                num_segments += 1
        # rank the segments in an accending order
        for key, value in episodes.items():
            episodes[key] = sorted(value, key=lambda x: int(x[0]))
        # re-split episodes according to time
        new_episodes = OrderedDict()
        MAX_TIME = 1000
        for eps, segs in episodes.items():
            start_time = -MAX_TIME
            working_ord = -1
            for part_id, file_path in segs:
                if int(part_id) - start_time >= MAX_TIME:
                    working_ord = part_id
                    new_episodes[f"{eps}-{working_ord}"] = []
                start_time = int(part_id)
                new_episodes[f"{eps}-{working_ord}"].append( (part_id, file_path) )
        episodes = new_episodes
        print(f'[Action] - num of episodes: {len(episodes)}, num of segments: {num_segments}') 
        return episodes


    def do_convert(self, 
                   eps_id: str, 
                   skip_frames: List[List[bool]], 
                   modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        """
        Converts action data for a given episode.

        Processes actions from .jsonl files, applies transformations, and handles frame skipping and remapping.

        :param eps_id: The ID of the episode.
        :type eps_id: str
        :param skip_frames: A list of lists of boolean flags indicating whether to skip frames.
        :type skip_frames: List[List[bool]]
        :param modal_file_path: A list of file paths for the modal data (action files).
        :type modal_file_path: List[Union[str, Path]]
        :returns: A tuple containing a list of keys (chunk start indices) and a list of pickled data values.
        :rtype: Tuple[List, List]
        """
        cache, keys, vals = [], [], []
        for _skip_frames, _modal_file_path in zip(skip_frames, modal_file_path):
            data = pickle.load(open(str(_modal_file_path), 'rb'))
            if _skip_frames is None:
                _skip_frames = ...
            if len(cache) == 0:
                cache = {k: v[_skip_frames] for k, v in data.items()}
            else:
                for k, v in data.items():
                    cache[k] = np.concatenate((cache[k], v[_skip_frames]), axis=0)

        for chunk_start in range(0, len(cache['attack']), self.chunk_size):
            chunk_end = chunk_start + self.chunk_size
            if chunk_end > len(cache['attack']):
                break
            val = {k: v[chunk_start:chunk_end] for k, v in cache.items()}
            keys.append(chunk_start)
            vals.append(pickle.dumps(val))
        
        return keys, vals


    def gen_frame_skip_flags(self, file_name: str) -> List[bool]:
        """
        Generates frame skip flags based on action data to identify no-operation frames.

        A frame is considered a no-op if the camera is static and all other actions are zero.

        :param file_name: The name of the action file (without extension).
        :type file_name: str
        :returns: A list of boolean flags, where True indicates the frame should be kept (not a no-op).
        :rtype: List[bool]
        """
        for dir in self.input_dirs:
            path = Path(dir) / f"{file_name}.pkl"
            if path.exists():
                break
        
        def _check_no_op(action: Dict):
            if np.any(action.pop('camera') != 0.):
                return True
            _sum = 0
            for key, val in action.items():
                _sum += val.sum()
            return _sum != np.array(0.)
        
        skip_flags = []
        with open(str(path), "rb") as f:
            action_pkl = pickle.load(f)
            traj_len = len(action_pkl['attack'])
            for fid in range(traj_len):
                f_action = {key: val[fid:fid+1] for key, val in action_pkl.items()}
                no_op_flag = _check_no_op(f_action)
                skip_flags.append(no_op_flag)
        
        return skip_flags


if __name__ == '__main__':
    """
    for debugging purpose
    """
    action_convert = ActionConvertCallback(
        input_dirs=[
            "/nfs-shared/data/contractors/all_9xx_Jun_29/actions"
        ], 
        chunk_size=32
    )
    episodes = action_convert.load_episodes()
    for idx, (eps, val) in enumerate(episodes.items()):
        print(eps, val)
        if idx > 5:
            break
    
    # test gen_frame_skip_flags
    action_path = val[-1][-1].stem
    skip_flags = action_convert.gen_frame_skip_flags(action_path)
    import ipdb; ipdb.set_trace()


