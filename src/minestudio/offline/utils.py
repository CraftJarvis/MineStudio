'''
Date: 2024-11-26 06:26:26
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-26 06:28:27
FilePath: /MineStudio/minestudio/train/utils.py
'''
from typing import Dict, Any, List
from omegaconf import DictConfig, ListConfig

def convert_to_normal(obj):
    """
    Recursively converts OmegaConf DictConfig and ListConfig objects to standard Python dicts and lists.

    This function is useful when working with configurations loaded by OmegaConf,
    as it allows you to convert them to native Python types for easier manipulation
    or serialization.

    :param obj: The object to convert. Can be a DictConfig, ListConfig, or any other type.
    :type obj: Any
    :returns: The converted object, with DictConfig and ListConfig instances replaced by dicts and lists respectively.
    :rtype: Any
    """
    if isinstance(obj, DictConfig) or isinstance(obj, Dict):
        return {key: convert_to_normal(value) for key, value in obj.items()}
    elif isinstance(obj, ListConfig) or isinstance(obj, List):
        return [convert_to_normal(item) for item in obj]
    else:
        return obj