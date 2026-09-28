'''
Date: 2024-12-14 01:46:36
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2024-12-14 02:00:17
FilePath: /MineStudio/minestudio/models/utils/download.py
'''
import huggingface_hub
import os
import pathlib

def download_model(model_name: str, local_dir: str = "downloads") -> str:
    """Downloads a specified model from Hugging Face Hub if it doesn't exist locally.

    Prompts the user for confirmation before downloading. The model is saved to a
    subdirectory within `local_dir` named after `model_name`.

    :param model_name: The name of the model to download. 
                       Valid names are "ROCKET-1", "VPT", "GROOT", "STEVE-1".
    :type model_name: str
    :param local_dir: The base directory to save downloaded models. Defaults to "downloads".
                      This will be relative to the parent directory of this script's location.
    :type local_dir: str
    :returns: The local path to the downloaded model directory if successful or if download was skipped
              by user choice but the directory was expected to exist (though it might not in that case).
              Returns None if download is skipped and directory doesn't exist, or if an error occurs.
    :rtype: str
    :raises AssertionError: if `model_name` is not one of the recognized names.
    """

    assert model_name in ["ROCKET-1", "VPT", "GROOT", "STEVE-1"], f"Unknown model: {model_name}"

    local_dir = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), local_dir, model_name)
    
    download = False
    response = input(f"Detecting missing {model_name}, do you want to download it from huggingface (Y/N)?\n")
    if not os.path.exists(local_dir):
        while True:
            if response == 'Y' or response == 'y':
                download = True
                break
            elif response == 'N' or response == 'n':
                break
            else:
                response = input("Please input Y or N:\n")

    if not download:
        return None

    print(f"Downloading {model_name} to {local_dir}")

    huggingface_hub.hf_hub_download(repo_id=f'CraftJarvis/{model_name}', filename='.', local_dir=local_dir, repo_type='model')
    
    return local_dir