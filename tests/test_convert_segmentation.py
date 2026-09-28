import ray
import argparse
from rich import print

from minestudio.data.minecraft.tools.convertion import ConvertManager
from minestudio.data.minecraft.callbacks import SegmentationConvertCallback, ActionConvertCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="lmdb chunk size")
    parser.add_argument("--segmentation-dir", type=str, required=False, nargs='+', 
                        help="directory containing segmentation files")
    parser.add_argument("--action-dir", type=str, required=False, nargs='+', 
                        help="directory containing action files")
    parser.add_argument("--output-dir", type=str, required=True, default="datasets",
                        help="directory saving lmdb files")
    parser.add_argument("--num-workers", default=4, type=int,  
                        help="the number of workers")
    args = parser.parse_args()
    
    ray.init()
    
    action_convert_kernel = ActionConvertCallback(
        input_dirs=args.action_dir, 
        chunk_size=args.chunk_size
    )
    
    segmentation_convert_kernel = SegmentationConvertCallback(
        input_dirs=args.segmentation_dir, 
        chunk_size=args.chunk_size
    )
    
    task_manager = ConvertManager(
        output_dir=args.output_dir,
        convert_kernel=segmentation_convert_kernel, 
        filter_kernel=action_convert_kernel,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )

    task_manager.prepare_tasks()
    task_manager.dispatch()
    