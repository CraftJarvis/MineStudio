from huggingface_hub import HfApi

if __name__ == '__main__':
    local_dir = "./simple"
    repo_id = "CraftJarvis/MineStudio_task_group.simple"
    api = HfApi()
    # api.create_repo(repo_id=repo_id, repo_type='dataset')
    # if not api.repo_exists(repo_id=repo_id, repo_type='dataset'):
    #     api.create_repo(repo_id=repo_id, repo_type='dataset')
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="dataset")