#from minestudio.online.utils.train.wandb_logger import WandbLogger
import ray

def get_current_session():
    try:
        return ray.get_actor("training_session")
    except ValueError:
        return None
    
def get_current_session_id():
    current_session = get_current_session()
    return ray.get(current_session.get_session_id.remote()) # type: ignore