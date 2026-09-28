'''
Date: 2024-12-06 16:42:49
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-06-06 13:59:32
FilePath: /MineStudio/minestudio/benchmark/utility/read_conf.py
'''
import os
import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal

from huggingface_hub import list_repo_files, hf_hub_download
from minestudio.utils import get_mine_studio_dir

def convert_yaml_to_callbacks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    commands = data.get('custom_init_commands', [])

    text = data.get('text', '')
    task_name = os.path.splitext(os.path.basename(yaml_file))[0]
    task_dict = {}
    task_dict['name'] = task_name
    task_dict['text'] = text

    return commands, task_dict

def prepare_task_configs(group_name: str, path: Optional[str] = None, refresh: bool = False) -> Dict:
    """
    group_name: str - used to specify the group name
    path: str - can be a local directory or a huggingface repo_id
    """
    root_dir = get_mine_studio_dir()
    local_dir = os.path.join(root_dir, "task_configs", group_name)
    if refresh and os.path.exists(local_dir):
        print(f"Refreshing the cache: removing existing task configs from: {local_dir}")
        shutil.rmtree(local_dir)
    if not os.path.exists(local_dir):
        if path is not None and os.path.isdir(path):
            shutil.copytree(path, local_dir)
        else:
            print(f"Downloading task configs from 🤗: {path}")
            all_files = list_repo_files(repo_id=path, repo_type='dataset')
            yaml_files = [f for f in all_files if f.endswith('.yaml')]
            for yaml_file in yaml_files:
                hf_hub_download(
                    repo_id=path, 
                    filename=yaml_file, 
                    local_dir=local_dir, 
                    repo_type='dataset'
                )
    yaml_files = { file_path.stem: str(file_path) for file_path in Path(local_dir).rglob("*.yaml") }
    return yaml_files