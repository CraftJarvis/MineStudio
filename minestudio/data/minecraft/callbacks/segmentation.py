'''
Date: 2025-01-09 05:42:00
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-17 15:26:38
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

    def create_from_config(config: Dict) -> 'SegmentationKernelCallback':
        return SegmentationKernelCallback(**config.get('segmentation', {}))

    def __init__(self, frame_width: int=224, frame_height: int=224):
        super().__init__()
        self.width = frame_width
        self.height = frame_height

    @property
    def name(self) -> str:
        return 'segmentation'

    def rle_encode(self, binary_mask):
        '''
        binary_mask: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = binary_mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            binary_mask[lo:hi] = 1
        return binary_mask.reshape(shape)

    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        if isinstance(dataset_paths[0], str):
            dataset_paths = [Path(path) for path in dataset_paths]
        action_paths = [path for path in dataset_paths if Path(path).stem in ['segment', 'segmentation']]
        return action_paths

    def do_decode(self, chunk: bytes, **kwargs) -> Dict:
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Dict:
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
                # import ipdb; ipdb.set_trace()
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
                    # print(f"look up: {last_key = }")
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
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, win_len: int, **kwargs) -> Tuple[Dict, np.ndarray]:
        traj_len = len(data['obj_id'])
        pad_data = dict()
        pad_data['event'] = data['event'] + [''] * (win_len - traj_len)
        pad_obj_id = np.zeros(win_len-traj_len, dtype=np.int32)
        pad_obj_mask = np.zeros((win_len-traj_len, self.height, self.width), dtype=np.uint8)
        pad_point = np.zeros((win_len-traj_len, 2), dtype=np.int32) - 1
        pad_frame_id = np.zeros(win_len-traj_len, dtype=np.int32) - 1
        pad_frame_range = np.zeros((win_len-traj_len, 2), dtype=np.int32) - 1
        if traj_len == 0:
            pad_data['obj_id'] = pad_obj_id
            pad_data['obj_mask'] = pad_obj_mask
            pad_data['point'] = pad_point
            pad_data['frame_id'] = pad_frame_id
            pad_data['frame_range'] = pad_frame_range
        else:
            pad_data['obj_id'] = np.concatenate([data['obj_id'], pad_obj_id], axis=0)
            pad_data['obj_mask'] = np.concatenate([data['obj_mask'], pad_obj_mask], axis=0)
            pad_data['point'] = np.concatenate([data['point'], pad_point], axis=0)
            pad_data['frame_id'] = np.concatenate([data['frame_id'], pad_frame_id], axis=0)
            pad_data['frame_range'] = np.concatenate([data['frame_range'], pad_frame_range], axis=0)
        pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len-traj_len, dtype=np.uint8)], axis=0)
        return pad_data, pad_mask

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

class SegmentationDrawFrameCallback(DrawFrameCallback):

    def __init__(self, 
                 start_point: Tuple[int, int]=(300, 10), 
                 draw_point: bool=True,
                 draw_mask: bool=True,
                 draw_event: bool=True,
                 draw_frame_id: bool=True,
                 draw_frame_range: bool=True):
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

    def load_episodes(self):
        
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
        The input video is connected end to end to form a complete trajectory, named eps_id. 
        However, the input data is processed independently, so its frame id is also independent. 
        When integrating it into a complete trajectory, the frame id needs to be remapped.
        That's why ``frame_id_re_mapping`` is used here, where ``ord`` indicates the part of the whole trajectory,
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
    