# Data Conversion

We provide tools to convert raw trajectory data into the MineStudio LMDB format. This conversion is crucial for efficient data loading and utilization within the MineStudio framework.

```{warning}
It is essential to perform the conversion to ensure that our data processing and model training pipelines can be effectively utilized.
```

## 1. Understanding the Conversion Process

The conversion process is managed by the `ConvertManager` class, which utilizes `ConvertWorker` (Ray remote actors) for parallel processing. Each modality (e.g., images, actions, metadata) in your raw dataset needs a corresponding `ModalConvertCallback` to handle its specific data reading and transformation logic.

**Key Components:**

*   **`ConvertManager`**: Orchestrates the conversion. It discovers raw data files, divides work among workers, and manages output.
*   **`ConvertWorker`**: A Ray actor that processes a subset of episodes for a given modality, converts the data using a `ModalConvertCallback`, and writes it to an LMDB file.
*   **`ModalConvertCallback`**: An abstract class that you need to implement for each data type you want to convert. MineStudio provides built-in callbacks for common types like images, actions, etc. (e.g., `ImageConvertCallback`, `ActionConvertCallback` from `minestudio.data.minecraft.callbacks.extension`). You will typically specify the path to your raw data within these callbacks.

## 2. Preparing Raw Trajectories

The `ConvertManager` and the specific `ModalConvertCallback` implementations will expect your raw data to be organized in a way they can discover. Typically, this means having separate directories for different data modalities or structured file naming.

For example, if you are converting video and action data:

*   **Video Data**: Might be a directory of `.mp4` files or image sequences.
    ```
    /path/to/your_raw_data/videos/
        episode_0001.mp4
        episode_0002.mp4
        ...
    ```
*   **Action Data**: Might be a directory of `.pkl`, `.json`, or `.csv` files.
    ```
    /path/to/your_raw_data/actions/
        episode_0001.pkl
        episode_0002.pkl
        ...
    ```

The exact structure and file types depend on the `ModalConvertCallback` you use or implement. Refer to the documentation of the specific callbacks for their expected input format.

## 3. Converting Trajectories to MineStudio LMDB Format

Instead of a command-line script, you use the `ConvertManager` API within a Python script. Here's how to convert a modality (e.g., actions):

```python
import ray
from minestudio.data.minecraft.tools.convertion import ConvertManager
# Import the specific ModalConvertCallback for the data type you are converting
# For example, for actions:
from minestudio.data.minecraft.callbacks.extension import ActionConvertCallback # Adjust import as per actual location

def main():
    # Initialize Ray (if not already initialized)
    if not ray.is_initialized():
        ray.init()

    # 1. Configure the ModalConvertCallback
    # This callback needs to know where your raw action files are.
    # The arguments for the callback will depend on its implementation.
    action_convert_kernel = ActionConvertCallback(
        source_dir='/path/to/your_raw_data/actions', # Path to raw action files
        # ... other parameters specific to ActionConvertCallback ...
    )

    # 2. (Optional) Configure a Filter Kernel
    # If you need to filter episodes or parts of episodes before conversion,
    # you can provide a filter_kernel. This is another ModalConvertCallback.
    # filter_kernel = YourFilterCallback(...)

    # 3. Initialize the ConvertManager
    convert_manager = ConvertManager(
        output_dir='/path/to/output/dataset/action', # Output directory for this modality's LMDB
        convert_kernel=action_convert_kernel,
        # filter_kernel=filter_kernel, # Uncomment if using a filter
        chunk_size=32,          # Affects how data is grouped in LMDB; also related to worker tasks
        num_workers=4           # Number of parallel ConvertWorker actors
    )

    # 4. Prepare tasks (discover and filter raw data)
    print("Preparing conversion tasks...")
    convert_manager.prepare_tasks()

    # 5. Dispatch tasks to workers for conversion
    print("Dispatching tasks to workers...")
    convert_manager.dispatch()

    print("Conversion complete.")
    ray.shutdown()

if __name__ == '__main__':
    main()
```

**Explanation:**

*   **`ModalConvertCallback` (e.g., `ActionConvertCallback`)**:
    *   You need to instantiate a specific callback for the modality you are converting.
    *   This callback is responsible for finding the raw data files (e.g., by looking in `source_dir`), reading them, and transforming them into the format expected for LMDB storage.
    *   The arguments to these callbacks (like `source_dir`) are crucial.
*   **`output_dir`**: This is the directory where the LMDB files for *this specific modality* will be stored. For example, if you are converting actions, point this to `/path/to/output/dataset/action`. For images, you'd run the conversion again with a different callback and a different `output_dir` like `/path/to/output/dataset/image`.
*   **`chunk_size`**: Influences how data is grouped.
*   **`num_workers`**: Determines the number of parallel worker processes. The `ConvertManager` divides the total number of episodes among these workers. Each worker will typically create its own LMDB file (or set of files) within the specified `output_dir`.

**To convert multiple modalities (e.g., video, actions, metadata):**

You would typically run the conversion process (steps 1-5 above) *multiple times*, once for each modality. Each run would use:
1.  A different `ModalConvertCallback` configured for that modality.
2.  A different `output_dir` specific to that modality (e.g., `../dataset/image`, `../dataset/action`).

## 4. Output File Structure

After running the conversion for each modality, your output directory will contain subdirectories for each modality, and within those, LMDB files generated by the `ConvertWorker`s.

If `output_dir` for an action conversion was `/path/to/output/dataset/action` and `num_workers=2`, the structure might look like:

```
/path/to/output/dataset/
├── action/
│   ├── shard_0/  # Data processed by worker 0
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── shard_1/  # Data processed by worker 1
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── meta.json # (or similar, if ConvertManager saves overall metadata)
├── image/        # After running conversion for images
│   ├── action-0/
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── shard_1/
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── meta.json
└── ... (other modalities)
```

```{note}
The exact naming of the sub-directories within the modality's `output_dir` (e.g., `shard_0`, `shard_1`) and the number of LMDB files will depend on the `ConvertManager`'s implementation details regarding how it assigns tasks to workers and how workers name their output. The example `meta.json` is illustrative; the manager might store overall metadata differently or not at all at the top level of the modality.
```

The `KernelManager` (used by `RawDataset` and `EventDataset`) is then configured with the parent directory (e.g., `/path/to/output/dataset/`) and will automatically discover these modality-specific LMDBs.