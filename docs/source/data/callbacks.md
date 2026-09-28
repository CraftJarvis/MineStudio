# Data Processing Callbacks

MineStudio employs a flexible callback mechanism to handle the loading, conversion, and visualization of data across different modalities. This design aims to achieve separation of concerns, decoupling data processing logic from the core data loading framework. Users can easily extend the system's functionality for custom raw data formats or new data modalities by implementing specific callback classes, without needing to modify the core code.

## 1. Design Philosophy

The core advantages of the callback mechanism are:

*   **Decoupling**: Separates the processing logic for specific modalities (e.g., decoding, transformation, augmentation, visualization) from generic data loaders (`RawDataset`, `EventDataset`) and data conversion tools (`ConvertManager`). This makes the core framework more versatile and stable.
*   **Extensibility**: Users can easily add support for new data modalities or custom data formats by simply implementing the corresponding callback interfaces.
*   **Customizability**: Users can tailor the processing of existing modalities to their specific needs, such as modifying data augmentation pipelines, changing how visual information is presented, or adjusting the details of data conversion.
*   **Code Reusability**: Common callback logic (like LMDB reading/writing) can be implemented in base classes, while specific modality callbacks focus on their unique processing tasks.

MineStudio defines three main base callback classes, serving runtime data loading, raw data format conversion, and data visualization respectively:

1.  `ModalKernelCallback`: Defines how to process specific modality data read from LMDB during data loading (e.g., within a `Dataset`'s `__getitem__`).
2.  `ModalConvertCallback`: Defines how to convert user's raw data files (e.g., `.mp4` videos, `.jsonl` action sequences) into the LMDB format used by MineStudio.
3.  `DrawFrameCallback`: Defines how to draw modality-specific information onto video frames during data visualization.

## 2. Detailed Explanation of Core Callback Types

The following provides a detailed introduction to these three core callback types and the key methods that need to be implemented.

### 2.1. `ModalKernelCallback`

`ModalKernelCallback` (defined in `minestudio.data.minecraft.callbacks.callback.ModalKernelCallback`) is used by `ModalKernel` and `KernelManager` during the data loading and processing pipeline. It is responsible for handling single data chunks or sequences of data chunks read from LMDB and transforming them into the format required for model training.

**Main Responsibilities:**

*   Decode raw byte data read from LMDB.
*   Merge multiple data chunks (if necessary).
*   Slice data according to a given time window and frame skipping parameters.
*   Pad data to meet fixed length requirements.
*   Perform data post-processing, such as data augmentation, format conversion, etc.

**Key Methods to Implement/Override:**

*   `__init__(self, read_bias: int = 0, win_bias: int = 0)`:
    *   Constructor. `read_bias` and `win_bias` are used to adjust the starting position of the window when reading data.
*   `name(self) -> str` (property):
    *   Returns the name of the modality handled by this callback (e.g., `"image"`, `"action"`). This is typically inferred automatically from the class name.
*   `filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]`:
    *   (Optional) Filters the list of dataset paths provided to `ModalKernel`. By default, it looks for subdirectories matching the modality name.
*   `do_decode(self, chunk: bytes, **kwargs) -> Any`:
    *   **[Core]** Decodes a single raw byte data chunk `chunk` read from LMDB into its original format (e.g., `np.ndarray` for images, `dict` for actions).
*   `do_merge(self, chunk_list: List[bytes], **kwargs) -> Union[List, Dict]`:
    *   **[Core]** Merges multiple decoded data chunks `chunk_list` into a single data structure. This is crucial for sequential data that spans multiple chunks.
*   `do_slice(self, data: Union[List, Dict], start: int, end: int, skip_frame: int, **kwargs) -> Union[List, Dict]`:
    *   **[Core]** Extracts a subsequence from the merged data `data` based on `start` (start frame index), `end` (end frame index), and `skip_frame` (frame skip count).
*   `do_pad(self, data: Union[List, Dict], pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Union[List, Dict], np.ndarray]`:
    *   **[Core]** If the sliced data length is less than `pad_len`, pads it at the position specified by `pad_pos` (`"left"` or `"right"`). Also returns a mask indicating which frames are valid and which are padded.
*   `do_postprocess(self, data: Dict, **kwargs) -> Dict`:
    *   (Optional) Performs post-processing on the finally processed data, such as applying data augmentations, converting to PyTorch tensors, etc.

### 2.2. `ModalConvertCallback`

`ModalConvertCallback` (defined in `minestudio.data.minecraft.callbacks.callback.ModalConvertCallback`) is used by `ConvertManager` and `ConvertWorker` during the data preprocessing stage. It is responsible for converting user-provided raw trajectory data (e.g., video files, action logs) into MineStudio's LMDB database format.

**Main Responsibilities:**

*   Discover and load raw data files from specified input directories.
*   Convert raw data file content into a sequence of byte chunks suitable for storage in LMDB.
*   (Optional) Generate frame skip flags for skipping certain frames during conversion.

**Key Methods to Implement/Override:**

*   `__init__(self, input_dirs: List[str], chunk_size: int)`:
    *   Constructor. `input_dirs` is a list of directories containing raw data files, and `chunk_size` defines the number of frames (or other units) contained in each LMDB data chunk.
*   `load_episodes(self) -> Dict[str, List[Tuple[str, str]]]`:
    *   **[Core]** Scans `self.input_dirs`, discovers all raw data files, and organizes them into a dictionary. The dictionary keys are episode IDs, and the values are lists of file paths (or other metadata) associated with that episode.
    *   The returned structure is typically `OrderedDict[eps_id, List[Tuple[modal_name, file_path]]]` or similar, indicating which modality files each episode contains.
*   `do_convert(self, eps_id: str, skip_frames: List[List[bool]], modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]`:
    *   **[Core]** Performs the actual conversion operation for a single episode (`eps_id`) and its corresponding raw file paths (`modal_file_path`).
    *   `skip_frames` is an optional list of frame skip flags.
    *   This method should read the raw files, process their content and split it into multiple data chunks (each corresponding to `chunk_size` frames), and then encode each chunk into a byte string.
    *   Returns a tuple containing `(key_list, chunk_list)`. `key_list` are the keys for each chunk (usually frame numbers or timestamps), and `chunk_list` are the corresponding encoded byte data chunks.
*   `gen_frame_skip_flags(self, file_name: str) -> List[bool]`:
    *   (Optional) Generates a boolean list for a given raw data file, indicating which frames should be skipped during conversion. Can return `None` or a list of all `False` if no frames need to be skipped.

### 2.3. `DrawFrameCallback`

`DrawFrameCallback` (defined in `minestudio.data.minecraft.callbacks.callback.DrawFrameCallback`) is used during data visualization to draw modality-specific information onto video frames. For example, displaying action data as text or overlaying segmentation masks on images.

**Main Responsibilities:**

*   Receive a batch of video frames and corresponding modality data.
*   Draw the modality data onto the corresponding video frames in graphical or textual form.

**Key Methods to Implement/Override:**

*   `draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int, **kwargs) -> np.ndarray`:
    *   **[Core]** This method receives a batch of video frames `frames` (usually a NumPy array of shape `(B, T, H, W, C)` or `(T, H, W, C)`, where B is batch size, T is sequence length) and a dictionary `infos` containing the corresponding modality data. `sample_idx` indicates the current sample in the batch.
    *   The keys of the `infos` dictionary are modality names (e.g., `"action"`, `"segmentation"`), and the values are the data for that modality.
    *   This method needs to iterate through the frames and corresponding `infos`, drawing the information onto the frames (e.g., using OpenCV drawing functions).
    *   Returns the video frames with the information drawn on them (NumPy array).

## 3. Examples of Built-in Callbacks

MineStudio provides a series of built-in callback implementations for common Minecraft data modalities. These implementations are located in the `minestudio.data.minecraft.callbacks` directory.

### 3.1. Image Callbacks

*   **`ImageKernelCallback`** (in `image.py`):
    *   **Purpose**: Processes image data loaded from LMDB.
    *   `do_decode`: Decodes byte strings into NumPy image arrays.
    *   `do_merge`: Concatenates a list of image chunks into a video sequence (NumPy array `(T, H, W, C)`).
    *   `do_slice`: Extracts a specified range of frames.
    *   `do_pad`: Pads the frame sequence.
    *   `do_postprocess`: Optionally applies `VideoAugmentation` for data augmentation and converts images to PyTorch tensors.
    *   **Parameters**: `frame_width`, `frame_height`, `enable_video_aug` (whether to enable video augmentation), `image_format` (e.g., "CHW", "HWC").

*   **`ImageConvertCallback`** (in `image.py`):
    *   **Purpose**: Converts raw video files (e.g., `.mp4`) or image sequence directories to LMDB format.
    *   `load_episodes`: Scans input directories for video files.
    *   `do_convert`: Uses OpenCV to read video frames, encoding every `chunk_size` frames into a JPEG byte string (or other format) as one LMDB data chunk.
    *   **Parameters**: `input_dirs`, `chunk_size`, `thread_pool` (number of threads for parallel video encoding).

### 3.2. Action Callbacks

*   **`ActionKernelCallback`** (in `action.py`):
    *   **Purpose**: Processes action data (usually a sequence of dictionaries) loaded from LMDB.
    *   `do_decode`: Decodes JSON-encoded byte strings into action dictionaries.
    *   `do_merge`: Merges a list of action dictionaries.
    *   `do_slice`: Extracts a specified range of actions.
    *   `do_pad`: Pads the action sequence (usually with zero actions or the last valid action).
    *   `do_postprocess`: Optionally includes the previous frame's action (`enable_prev_action`).
    *   **Parameters**: `enable_prev_action`, `prev_action_pad_val`, `read_bias`, `win_bias`.

*   **`VectorActionKernelCallback`** (in `action.py`, inherits from `ActionKernelCallback`):
    *   **Purpose**: Specifically handles vectorized action representations.
    *   `do_postprocess`: Builds upon `ActionKernelCallback` to convert dictionary-form actions into fixed-dimension vectors, or vice-versa.
    *   Provides `vector_to_action` and `action_to_vector` methods.
    *   **Parameters**: `action_chunk_size` (similar to `chunk_size`, but specific to context length for action vectorization), `return_type` ("vector" or "dict").

*   **`ActionDrawFrameCallback`** (in `action.py`):
    *   **Purpose**: Draws textual representations of action information onto video frames.
    *   `draw_frames`: Iterates through action data, formats action key-value pairs for each timestep into strings, and draws them onto frames at specified positions using OpenCV's `putText`.
    *   **Parameters**: `start_point` (starting coordinates for text drawing).

*   **`ActionConvertCallback`** (in `action.py`):
    *   **Purpose**: Converts raw action files (typically `.jsonl` files, where each line is a JSON object representing one frame's actions) to LMDB.
    *   `load_episodes`: Scans input directories for `.jsonl` files.
    *   `do_convert`: Reads `.jsonl` files, and every `chunk_size` actions are JSON-encoded and stored as one LMDB data chunk.
    *   **Parameters**: `input_dirs`, `chunk_size`, `action_transformer_kwargs` (parameters for initializing `ActionTransformer`, allowing action transformation and filtering).

### 3.3. MetaInfo Callbacks

*   **`MetaInfoKernelCallback`** (in `meta_info.py`):
    *   **Purpose**: Processes game metadata, such as player position, health, time, etc.
    *   `do_decode`: Decodes JSON-encoded byte strings into metainfo dictionaries.
    *   `do_merge`, `do_slice`, `do_pad`: Similar to `ActionKernelCallback` for handling sequences of dictionaries.
    *   **Parameters**: No special constructor parameters, relies on the base class.

*   **`MetaInfoDrawFrameCallback`** (in `meta_info.py`):
    *   **Purpose**: Draws metainfo onto video frames.
    *   `draw_frames`: Formats specific key-value pairs from the metainfo dictionary and draws them onto frames.
    *   **Parameters**: `start_point`.

*   **`MetaInfoConvertCallback`** (in `meta_info.py`):
    *   **Purpose**: Converts raw metainfo files (typically `.jsonl` or `.pkl` files containing metainfo dictionaries) to LMDB.
    *   `load_episodes`: Scans input directories for metainfo files.
    *   `do_convert`: Reads files, and every `chunk_size` frames of metainfo are JSON-encoded and stored as one LMDB data chunk.
    *   **Parameters**: `input_dirs`, `chunk_size`.

### 3.4. Segmentation Callbacks

*   **`SegmentationKernelCallback`** (in `segmentation.py`):
    *   **Purpose**: Processes image segmentation mask data.
    *   `do_decode`: Decodes byte strings (possibly RLE-encoded masks or other compressed formats) into segmentation masks (NumPy arrays).
    *   `do_merge`, `do_slice`, `do_pad`: Performs corresponding processing on sequences of segmentation masks.
    *   `do_postprocess`: May include resizing masks, remapping class IDs, etc.
    *   **Parameters**: `frame_width`, `frame_height` (target frame dimensions), `seg_re_map` (class ID remapping dictionary).

*   **`SegmentationDrawFrameCallback`** (in `segmentation.py`):
    *   **Purpose**: Overlays segmentation masks or related information (like target points) onto video frames.
    *   `draw_frames`: Can draw segmentation masks with different colors or highlight specific objects.
    *   **Parameters**: `start_point`, `draw_point`, `draw_mask`, `draw_event`, `draw_frame_id`, `draw_frame_range`, and a color list `COLORS`.

*   **`SegmentationConvertCallback`** (in `segmentation.py`):
    *   **Purpose**: Converts raw segmentation data files (e.g., `.pkl` files containing RLE-encoded masks) to LMDB.
    *   `load_episodes`: Scans input directories for segmentation data files.
    *   `do_convert`: Reads raw segmentation data, encodes it (if necessary), and stores every `chunk_size` frames of segmentation data as one LMDB data chunk.
    *   **Parameters**: `input_dirs`, `chunk_size`.

Through these callbacks, MineStudio provides a powerful and flexible framework for handling various Minecraft data. Users can modify these built-in callbacks or create their own callback implementations for entirely new data types.
