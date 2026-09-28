'''
Date: 2025-01-09 05:42:00
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-21 22:31:03
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/segmentation.py
'''
import cv2
import random
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, DrawFrameCallback, ModalConvertCallback
from minestudio.utils.register import Registers

SEG_RE_MAP = {
    0: 0, 1: 3, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
}

@Registers.modal_kernel_callback.register
class SegmentationKernelCallback(ModalKernelCallback):
    """
    Callback for handling segmentation data kernels.

    This callback processes segmentation masks, including RLE encoding/decoding,
    filtering dataset paths, merging data chunks, slicing, and padding.
    """

    def create_from_config(config: Dict) -> 'SegmentationKernelCallback':
        """
        Create a SegmentationKernelCallback instance from a configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict
        :returns: An instance of SegmentationKernelCallback.
        :rtype: SegmentationKernelCallback
        """
        return SegmentationKernelCallback(**config.get('segmentation', {}))

    def __init__(self, frame_width: int=224, frame_height: int=224):
        """
        Initialize SegmentationKernelCallback.

        :param frame_width: The width of the frame.
        :type frame_width: int
        :param frame_height: The height of the frame.
        :type frame_height: int
        """
        super().__init__()
        self.width = frame_width
        self.height = frame_height

    @property
    def name(self) -> str:
        """
        Get the name of the callback.

        :returns: The name 'segmentation'.
        :rtype: str
        """
        return 'segmentation'

    def rle_encode(self, binary_mask: np.ndarray) -> str:
        """
        Encode a binary mask using run-length encoding (RLE).

        :param binary_mask: Numpy array, 1 - mask, 0 - background.
        :type binary_mask: np.ndarray
        :returns: Run length as a string.
        :rtype: str
        """
        pixels = binary_mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self, mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode a run-length encoded (RLE) mask.

        :param mask_rle: Run-length as a string (start length).
        :type mask_rle: str
        :param shape: (height, width) of the array to return.
        :type shape: Tuple[int, int]
        :returns: Numpy array, 1 - mask, 0 - background.
        :rtype: np.ndarray
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            binary_mask[lo:hi] = 1
        return binary_mask.reshape(shape)

    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        """
        Filter dataset paths to include only segmentation-related files.

        :param dataset_paths: A list of dataset paths (strings or Path objects).
        :type dataset_paths: List[Union[str, Path]]
        :returns: A list of Path objects pointing to segmentation files.
        :rtype: List[Path]
        """
        if isinstance(dataset_paths[0], str):
            dataset_paths = [Path(path) for path in dataset_paths]
        action_paths = [path for path in dataset_paths if Path(path).stem in ['segment', 'segmentation']]
        return action_paths

    def do_decode(self, chunk: bytes, **kwargs) -> Dict:
        """
        Decode a data chunk using pickle.

        :param chunk: The data chunk to decode.
        :type chunk: bytes
        :param kwargs: Additional keyword arguments.
        :returns: The decoded data as a dictionary.
        :rtype: Dict
        """
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Dict:
        """
        Merge a list of data chunks into a single dictionary.

        Handles object ID remapping, mask resizing, and event processing.

        :param chunk_list: A list of data chunks (bytes).
        :type chunk_list: List[bytes]
        :param kwargs: Additional keyword arguments, e.g., 'event_constrain'.
        :returns: A dictionary containing the merged segmentation data.
        :rtype: Dict
        """
        raw_content = []
        for chunk_bytes in chunk_list:
            raw_content += self.do_decode(chunk_bytes)

        nb_frames = len(raw_content)
        res_content = {
            "obj_id": [-1 for _ in range(nb_frames)],
            "obj_mask": [np.zeros((self.height, self.width), dtype=np.uint8) for _ in range(nb_frames)], 
            "event": ["" for _ in range(nb_frames)],
            "point": [np.array((-1, -1)) for _ in range(nb_frames)],
            "frame_id": [-1 for _ in range(nb_frames)],
            "frame_range": [np.array((-1, -1)) for _ in range(nb_frames)],
        }

        last_key = None
        for wid in range(len(raw_content)-1, -1, -1):
            if len(raw_content[wid]) == 0:
                continue
            if last_key is None or last_key not in raw_content[wid]:
                # the start of a new interaction
                if kwargs.get('event_constrain', None) is None:
                    last_key = random.choice(list(raw_content[wid].keys())) #! random pick one
                else:
                    last_key = None
                    for lookup_key in raw_content[wid]:
                        if lookup_key[-1].replace("minecraft.", "") == kwargs['event_constrain']:
                            last_key = lookup_key
                            break
                    if last_key is None:
                        continue
                last_event = raw_content[wid][last_key]["event"]
            # during an interaction, `last_key` denotes the selected interaction
            frame_content = raw_content[wid][last_key]
            res_content["obj_id"][wid] = SEG_RE_MAP[ frame_content["obj_id"] ]
            obj_mask = self.rle_decode(frame_content["rle_mask"], (360, 640))
            res_content["obj_mask"][wid] = cv2.resize(obj_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            res_content["event"][wid] = frame_content["event"]
            if frame_content["point"] is not None:
                res_content["point"][wid] = np.array(frame_content["point"]) / np.array([360., 640.]) # normalize to [0, 1]
            res_content["frame_id"][wid] = frame_content["frame_id"]
            res_content["frame_range"][wid] = np.array(frame_content["frame_range"])

        for key in res_content:
            if key == 'event':
                continue
            res_content[key] = np.array(res_content[key])

        return res_content

    def do_slice(self, data: Dict, start: int, end: int, skip_frame: int, **kwargs) -> Dict:
        """
        Slice the data based on start, end, and skip_frame parameters.

        :param data: The input data dictionary.
        :type data: Dict
        :param start: The starting index for slicing.
        :type start: int
        :param end: The ending index for slicing.
        :type end: int
        :param skip_frame: The step for slicing (frame skipping).
        :type skip_frame: int
        :param kwargs: Additional keyword arguments.
        :returns: A dictionary containing the sliced data.
        :rtype: Dict
        """
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Dict, np.ndarray]:
        """
        Pad the data to a specified length.

        :param data: The input data dictionary.
        :type data: Dict
        :param pad_len: The length of the padding to add.
        :type pad_len: int
        :param pad_pos: The position to add padding ('left' or 'right').
        :type pad_pos: Literal["left", "right"]
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing the padded data dictionary and the padding mask.
        :rtype: Tuple[Dict, np.ndarray]
        """
        traj_len = len(data['obj_id'])
        pad_data = dict()
        pad_obj_id = np.zeros(pad_len, dtype=np.int32)
        pad_obj_mask = np.zeros((pad_len, self.height, self.width), dtype=np.uint8)
        pad_point = np.zeros((pad_len, 2), dtype=np.int32) - 1
        pad_frame_id = np.zeros(pad_len, dtype=np.int32) - 1
        pad_frame_range = np.zeros((pad_len, 2), dtype=np.int32) - 1
        if traj_len == 0:
            pad_data['obj_id'] = pad_obj_id
            pad_data['obj_mask'] = pad_obj_mask
            pad_data['point'] = pad_point
            pad_data['frame_id'] = pad_frame_id
            pad_data['frame_range'] = pad_frame_range
        else:
            if pad_pos == "left":
                pad_data['event'] = [''] * pad_len + data['event']
                pad_data['obj_id'] = np.concatenate([pad_obj_id, data['obj_id']], axis=0)
                pad_data['obj_mask'] = np.concatenate([pad_obj_mask, data['obj_mask']], axis=0)
                pad_data['point'] = np.concatenate([pad_point, data['point']], axis=0)
                pad_data['frame_id'] = np.concatenate([pad_frame_id, data['frame_id']], axis=0)
                pad_data['frame_range'] = np.concatenate([pad_frame_range, data['frame_range']], axis=0)
                pad_mask = np.concatenate([np.zeros(pad_len, dtype=np.uint8), np.ones(traj_len, dtype=np.uint8)], axis=0)
            elif pad_pos == "right":
                pad_data['event'] = data['event'] + [''] * pad_len
                pad_data['obj_id'] = np.concatenate([data['obj_id'], pad_obj_id], axis=0)
                pad_data['obj_mask'] = np.concatenate([data['obj_mask'], pad_obj_mask], axis=0)
                pad_data['point'] = np.concatenate([data['point'], pad_point], axis=0)
                pad_data['frame_id'] = np.concatenate([data['frame_id'], pad_frame_id], axis=0)
                pad_data['frame_range'] = np.concatenate([data['frame_range'], pad_frame_range], axis=0)
                pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(pad_len, dtype=np.uint8)], axis=0)
        return pad_data, pad_mask

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

class SegmentationDrawFrameCallback(DrawFrameCallback):
    """
    Callback for drawing segmentation information on frames.

    This callback can draw points, masks, event text, frame IDs, and frame ranges
    onto video frames.
    """

    def __init__(self,
                 start_point: Tuple[int, int]=(300, 10),
                 draw_point: bool=True,
                 draw_mask: bool=True,
                 draw_event: bool=True,
                 draw_frame_id: bool=True,
                 draw_frame_range: bool=True):
        """
        Initialize SegmentationDrawFrameCallback.

        :param start_point: The (x, y) starting coordinates for drawing text.
        :type start_point: Tuple[int, int]
        :param draw_point: Whether to draw the interaction point.
        :type draw_point: bool
        :param draw_mask: Whether to draw the object mask.
        :type draw_mask: bool
        :param draw_event: Whether to draw the event text.
        :type draw_event: bool
        :param draw_frame_id: Whether to draw the frame ID.
        :type draw_frame_id: bool
        :param draw_frame_range: Whether to draw the frame range.
        :type draw_frame_range: bool
        """
        super().__init__()
        self.x, self.y = start_point
        self.draw_point = draw_point
        self.draw_mask = draw_mask
        self.draw_event = draw_event
        self.draw_frame_id = draw_frame_id
        self.draw_frame_range = draw_frame_range

    def draw_frame(self,
                   frame: np.ndarray,
                   point: Optional[Tuple[int, int]]=None,
                   obj_mask: Optional[np.ndarray]=None,
                   obj_id: Optional[int]=None,
                   event: Optional[str]=None,
                   frame_id: Optional[int]=None,
                   frame_range: Optional[Tuple[int, int]]=None) -> np.ndarray:
        """
        Draw segmentation information on a single frame.

        :param frame: The input frame (numpy array).
        :type frame: np.ndarray
        :param point: The (y, x) coordinates of the interaction point (normalized).
        :type point: Optional[Tuple[int, int]]
        :param obj_mask: The object mask (numpy array).
        :type obj_mask: Optional[np.ndarray]
        :param obj_id: The object ID.
        :type obj_id: Optional[int]
        :param event: The event string.
        :type event: Optional[str]
        :param frame_id: The frame ID.
        :type frame_id: Optional[int]
        :param frame_range: The frame range tuple.
        :type frame_range: Optional[Tuple[int, int]]
        :returns: The frame with drawn segmentation information.
        :rtype: np.ndarray
        """
        frame = frame.copy()
        if self.draw_point and point is not None and point[0] != -1:
            x, y = point
            x = int(x * frame.shape[1])
            y = int(y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        if self.draw_mask and obj_id is not None and obj_id != -1 and obj_mask is not None:
            colors = np.array(COLORS[obj_id]).reshape(1, 1, 3)
            obj_mask = (obj_mask[..., None] * colors).astype(np.uint8)
            obj_mask = obj_mask[:, :, ::-1] # bgr -> rgb
            frame = cv2.addWeighted(frame, 1.0, obj_mask, 0.5, 0.0)
            cv2.putText(frame, f"Mask Area: {obj_mask.sum()}", (self.x+10, self.y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.draw_event and event is not None:
            cv2.putText(frame, f"Event: {event}", (self.x+10, self.y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.draw_frame_id and frame_id is not None:
            cv2.putText(frame, f"Frame ID: {frame_id}", (self.x+10, self.y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.draw_frame_range and frame_range is not None:
            cv2.putText(frame, f"Frame Range: {frame_range}", (self.x+10, self.y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int) -> np.ndarray:
        """
        Draw segmentation information on a list of frames.

        :param frames: A list of frames or a numpy array of frames.
        :type frames: Union[np.ndarray, List]
        :param infos: A dictionary containing segmentation information.
        :type infos: Dict
        :param sample_idx: The index of the sample to process.
        :type sample_idx: int
        :returns: A list of frames with drawn segmentation information.
        :rtype: List[np.ndarray]
        """
        cache_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            frame_info = infos['segmentation']
            obj_id = frame_info['obj_id'][sample_idx][frame_idx].item()
            obj_mask = frame_info['obj_mask'][sample_idx][frame_idx]
            point = (frame_info['point'][sample_idx][frame_idx][1].item(), frame_info['point'][sample_idx][frame_idx][0].item())
            if isinstance(obj_mask, torch.Tensor):
                obj_mask = obj_mask.numpy()
            event = frame_info['event'][sample_idx][frame_idx]
            frame_id = frame_info['frame_id'][sample_idx][frame_idx].item()
            frame_range = frame_info['frame_range'][sample_idx][frame_idx].numpy()
            frame = self.draw_frame(frame, point, obj_mask, obj_id, event, frame_id, frame_range)
            cache_frames.append(frame)
        return cache_frames

import re
from rich import print
from tqdm import tqdm
from collections import OrderedDict


class SegmentationConvertCallback(ModalConvertCallback):
    """
    Callback for converting segmentation data from raw RLE files.

    This callback loads episodes from directories containing RLE files,
    processes them, and converts them into a structured format.
    """

    def load_episodes(self) -> OrderedDict:
        """
        Load episodes from input directories containing RLE files.

        Identifies episode segments, sorts them, and re-splits them based on time.

        :returns: An OrderedDict where keys are episode IDs and values are lists of segment file paths.
        :rtype: OrderedDict
        """
        
        CONTRACTOR_PATTERN = r"^(.*?)-(\d+)$"
        
        episodes = OrderedDict()
        num_segments = 0
        for source_dir in self.input_dirs:
            print("Current input directory: ", source_dir) # action file ends with `.pkl`
            for file_path in tqdm(Path(source_dir).rglob("*.rle"), desc="Looking for source files"):
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
        print(f'[Segmentation] - num of episodes: {len(episodes)}, num of segments: {num_segments}') 
        return episodes


    def do_convert(self,
                   eps_id: str,
                   skip_frames: List[List[bool]],
                   modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        """
        Convert segmentation data for a given episode.

        The input video is connected end to end to form a complete trajectory, named eps_id.
        However, the input data is processed independently, so its frame id is also independent.
        When integrating it into a complete trajectory, the frame id needs to be remapped.
        That's why ``frame_id_re_mapping`` is used here, where ``ord`` indicates the part of the whole trajectory.

        :param eps_id: The ID of the episode.
        :type eps_id: str
        :param skip_frames: A list of lists of boolean flags indicating whether to skip frames.
        :type skip_frames: List[List[bool]]
        :param modal_file_path: A list of file paths for the modal data.
        :type modal_file_path: List[Union[str, Path]]
        :returns: A tuple containing a list of keys (chunk start indices) and a list of pickled data values.
        :rtype: Tuple[List, List]
        """
        cache, keys, vals = [], [], []
        frame_id_re_mapping = dict()
        new_frame_counter = 0
        for ord, (_skip_frames, _modal_file_path) in enumerate(zip(skip_frames, modal_file_path)):
            data = pickle.load(open(str(_modal_file_path), 'rb'))
            for ori_frame_id, skip_flag in enumerate(_skip_frames):
                if not skip_flag:
                    continue
                frame_id_re_mapping[(ord, ori_frame_id)] = new_frame_counter
                new_frame_counter += 1

            for ori_frame_id, skip_flag in enumerate(_skip_frames):
                if not skip_flag:
                    continue

                frame_content = {}
                for k in data['video_annos'].get(ori_frame_id, []):
                    inter_key = (k[0], k[1])
                    ori_frame_range = data['rle_mask_mapping'][k]["frame_range"]
                    remaining_event_frames = [
                        frame_id_re_mapping[(ord, ori_frame_id)] 
                            for ori_frame_id in range(ori_frame_range[0], ori_frame_range[1]+1) 
                                if (ord, ori_frame_id) in frame_id_re_mapping
                    ]
                    if len(remaining_event_frames) == 0:
                        new_frame_range = (-1, -1)
                    else:
                        new_frame_range = (min(remaining_event_frames), max(remaining_event_frames))
                    inter_val = {
                        "obj_id": data["rle_mask_mapping"][k]["obj_id"],        # type of the interaction
                        "rle_mask": data["rle_mask_mapping"][k]["rle_mask"],    # run-length encoding of the object mask
                        "event": data["rle_mask_mapping"][k]["event"],          # event description of an interaction
                        "point": data["rle_mask_mapping"][k]["point"],          # centroid of the object mask
                        "ori_frame_id": ori_frame_id,                           # generally not used
                        "ori_frame_range": ori_frame_range,                     # generally not used
                        "frame_id": frame_id_re_mapping[(ord, ori_frame_id)],   # use the order w.r.t. the whole trajectory
                        "frame_range": new_frame_range,                         # use the order w.r.t. the whole trajectory
                    }
                    frame_content[inter_key] = inter_val
                cache.append(frame_content)

        for chunk_start in range(0, len(cache), self.chunk_size):
            chunk_end = chunk_start + self.chunk_size
            if chunk_end > len(cache):
                break
            val = cache[chunk_start:chunk_end]
            keys.append(chunk_start)
            vals.append(pickle.dumps(val))

        return keys, vals


if __name__ == '__main__':
    """
    for debugging purpose
    """
    segmentation_convert = SegmentationConvertCallback(
        input_dirs=[
            "/nfs-shared-2/shaofei/contractor_segment_new/9xx"
        ], 
        chunk_size=32
    )
    episodes = segmentation_convert.load_episodes()
    for idx, (key, val) in enumerate(episodes.items()):
        print(key, val)
        if idx > 5:
            break
    import ipdb; ipdb.set_trace()
