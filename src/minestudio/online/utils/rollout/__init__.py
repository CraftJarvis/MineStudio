import ray

def get_rollout_manager():
    rollout_manager = None
    try:
        rollout_manager = ray.get_actor("rollout_manager")
    except ValueError:
        pass
    print(rollout_manager)
    return rollout_manager