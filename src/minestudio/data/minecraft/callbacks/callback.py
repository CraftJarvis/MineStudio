'''
Date: 2025-01-09 05:08:19
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-21 22:28:09
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/callback.py
'''
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

class ModalConvertCallback:
    """
    Base class for callbacks that convert raw trajectory data into MineStudio's built-in format.

    Users should implement the methods of this class to define how their specific 
    data format is converted.
    """

    def __init__(self, input_dirs: List[str], chunk_size: int):
        """
        Initializes the ModalConvertCallback.

        :param input_dirs: A list of directory paths containing the raw input data.
        :type input_dirs: List[str]
        :param chunk_size: The size of data chunks to be processed at a time.
        :type chunk_size: int
        """
        self.input_dirs = input_dirs
        self.chunk_size = chunk_size

    def load_episodes(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Identifies and loads raw data from the specified input directories.

        This method should be implemented by the user to parse their data structure 
        and return a dictionary mapping episode names to a list of file parts.

        :returns: A dictionary where keys are episode names and values are lists of 
                  tuples. Each tuple contains a part identifier and the corresponding 
                  file path for that part of the episode.
        :rtype: Dict[str, List[Tuple[str, str]]]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

    def do_convert(self, eps_id: str, skip_frames: List[List[bool]], modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        """
        Converts the raw modal data for a given episode into the desired format.

        `skip_frames` and `modal_file_path` are aligned lists, indicating which frames 
        to skip in each corresponding file part.

        :param eps_id: The identifier of the episode being converted.
        :type eps_id: str
        :param skip_frames: A list of lists of boolean flags. Each inner list corresponds 
                            to a file part in `modal_file_path` and indicates which 
                            frames within that part should be skipped.
        :type skip_frames: List[List[bool]]
        :param modal_file_path: A list of file paths (or Path objects) for the modal data files 
                                or parts of files that constitute the episode.
        :type modal_file_path: List[Union[str, Path]]
        :returns: A tuple containing two lists: chunk keys and chunk values, representing 
                  the converted data.
        :rtype: Tuple[List, List]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

    def gen_frame_skip_flags(self, file_name: str) -> List[bool]:
        """
        Generates a list of boolean flags indicating which frames to skip in a given file.

        This method should be implemented if users want to filter out specific frames 
        based on the content or metadata of this modality.

        :param file_name: The name of the file for which to generate skip flags.
        :type file_name: str
        :returns: A list of boolean flags. True indicates the frame should be skipped.
        :rtype: List[bool]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError


class ModalKernelCallback:
    """
    Base class for callbacks that define how a specific modality of data is handled 
    within the ModalKernel.

    Users must implement this callback for their custom modal data to manage operations 
    like decoding, merging, slicing, and padding.
    """

    @staticmethod
    def create_from_config(config: Dict) -> 'ModalKernelCallback':
        """
        Factory method to create a ModalKernelCallback instance from a configuration dictionary.

        :param config: A dictionary containing the configuration parameters for the callback.
        :type config: Dict
        :returns: An instance of a ModalKernelCallback subclass.
        :rtype: ModalKernelCallback
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

    def __init__(self, read_bias: int=0, win_bias: int=0):
        """
        Initializes the ModalKernelCallback.

        :param read_bias: An integer bias applied when reading data. Defaults to 0.
        :type read_bias: int, optional
        :param win_bias: An integer bias applied to the window length. Defaults to 0.
        :type win_bias: int, optional
        """
        self.read_bias = read_bias
        self.win_bias = win_bias

    @property
    def name(self) -> str:
        """
        Returns the name of the modality this callback handles (e.g., "image", "action").

        :returns: The name of the modality.
        :rtype: str
        :raises NotImplementedError: If this property is not implemented by a subclass.
        """
        raise NotImplementedError
    
    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        """
        Filters a list of potential dataset paths to select those relevant to this modality.

        `dataset_paths` contains all possible paths pointing to different LMDB folders. 
        This method should be implemented to identify and return only the paths 
        that contain data for the modality handled by this callback.

        :param dataset_paths: A list of potential dataset directory paths (strings or Path objects).
        :type dataset_paths: List[Union[str, Path]]
        :returns: A filtered list of Path objects pointing to the relevant LMDB datasets.
        :rtype: List[Path]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError
    
    def do_decode(self, chunk: bytes, **kwargs) -> Any:
        """
        Decodes a raw byte chunk of modal data into its usable format.

        Data is stored in LMDB files as byte chunks. Since decoding methods vary 
        for different modalities, users must implement this method to specify how 
        their data should be decoded.

        :param chunk: The raw byte string (chunk) to be decoded.
        :type chunk: bytes
        :param \\**kwargs: Additional keyword arguments that might be needed for decoding.
        :returns: The decoded data in its appropriate format (e.g., a NumPy array for images).
        :rtype: Any
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError
    
    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Union[List, Dict]:
        """
        Merges a list of decoded data chunks into a continuous sequence or structure.

        When reading a long trajectory segment, the system automatically reads and decodes 
        multiple chunks. This method defines how these decoded chunks for a specific 
        modality are combined into a single, coherent data sequence (e.g., a list of frames 
        or a dictionary of features).

        :param chunk_list: A list of decoded data chunks.
        :type chunk_list: List[bytes]
        :param \\**kwargs: Additional keyword arguments that might be needed for merging.
        :returns: The merged data, typically a list or dictionary representing the continuous sequence.
        :rtype: Union[List, Dict]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

    def do_slice(self, data: Union[List, Dict], start: int, end: int, skip_frame: int, **kwargs) -> Union[List, Dict]:
        """
        Extracts a slice from the modal data based on start, end, and skip_frame parameters.

        Since data formats can vary significantly between modalities, users need to 
        implement this method to define how slicing operations are performed on their specific data type.

        :param data: The modal data (e.g., a list of frames, a dictionary of features) to be sliced.
        :type data: Union[List, Dict]
        :param start: The starting index for the slice (inclusive).
        :type start: int
        :param end: The ending index for the slice (inclusive).
        :type end: int
        :param skip_frame: The interval at which to select frames/data points within the slice.
        :type skip_frame: int
        :param \\**kwargs: Additional keyword arguments that might be needed for slicing.
        :returns: The sliced portion of the data.
        :rtype: Union[List, Dict]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

    def do_pad(self, data: Union[List, Dict], pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Union[List, Dict], np.ndarray]:
        """
        Pads the modal data to a specified length if it's shorter than required.

        Users need to implement this method to define how padding is applied 
        (e.g., repeating the last frame, adding zero vectors) and to return a mask 
        indicating the padded elements.

        :param data: The modal data to be padded.
        :type data: Union[List, Dict]
        :param pad_len: The number of elements to add as padding.
        :type pad_len: int
        :param pad_pos: The position where padding should be added, either "left" or "right".
        :type pad_pos: Literal["left", "right"]
        :param \\**kwargs: Additional keyword arguments that might be needed for padding.
        :returns: A tuple containing the padded data and a NumPy array representing the padding mask 
                  (1 for original data, 0 for padded data).
        :rtype: Tuple[Union[List, Dict], np.ndarray]
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

    def do_postprocess(self, data: Dict, **kwargs) -> Dict:
        """
        Performs optional post-processing operations on the sampled modal data.

        This method can be used for tasks like data augmentation or further transformations 
        after the data has been read, sliced, and padded.

        :param data: A dictionary containing the modal data (and potentially its mask) to be post-processed.
        :type data: Dict
        :param \\**kwargs: Additional keyword arguments that might be needed for post-processing.
        :returns: The post-processed data dictionary.
        :rtype: Dict
        """
        return data


class DrawFrameCallback:
    """
    Base class for callbacks that draw overlay information onto video frames for visualization purposes.
    """
    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int, **kwargs) -> np.ndarray:
        """
        Draws specified information onto a set of video frames.

        This method needs to be implemented by users who want to visualize datasets 
        with custom overlays (e.g., drawing actions, metadata) on the video frames.

        :param frames: A list of frames (e.g., Pillow images) or a NumPy array of frames (T, H, W, C).
        :type frames: Union[np.ndarray, List]
        :param infos: A dictionary containing the information to be drawn on the frames. 
                      The structure of this dictionary depends on what the user wants to display.
        :type infos: Dict
        :param sample_idx: The index of the current sample being processed/visualized.
        :type sample_idx: int
        :param \\**kwargs: Additional keyword arguments that might be needed for drawing.
        :returns: A NumPy array of frames (T, H, W, C) with the information drawn on them.
        :rtype: np.ndarray
        :raises NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError

