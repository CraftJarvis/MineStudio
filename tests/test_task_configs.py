from rich import print
from minestudio.benchmark import prepare_task_configs
from minestudio.simulator.callbacks import load_callbacks_from_config

if __name__ == '__main__':
    name_file_mapping = prepare_task_configs("simple", path="CraftJarvis/MineStudio_task_group.simple", refresh=False)
    callbacks = load_callbacks_from_config(name_file_mapping["collect_wood"])
    print(callbacks)