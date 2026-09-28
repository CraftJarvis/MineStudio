import psutil
from minestudio.simulator.minerl.env.database_manager import collect_garbage, get_instance_num
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_pid', type=int)
    args = parser.parse_args()
    parent_pid = args.parent_pid

    count_after_parent_exit = 0
    while True:
        collect_garbage()
        if not psutil.pid_exists(parent_pid):
            if get_instance_num() == 0:
                break
            count_after_parent_exit += 1
            if count_after_parent_exit > 10:
                break
        time.sleep(10)