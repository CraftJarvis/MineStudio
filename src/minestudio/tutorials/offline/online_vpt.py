from numpy import roll
from omegaconf import OmegaConf
import hydra
import logging
from jarvis.arm.online.rollout.rollout_manager import RolloutManager
from jarvis.arm.online.utils.rollout import get_rollout_manager
import jarvis.arm.online.utils.registry as registry
from omegaconf import DictConfig
import ray
import wandb
import uuid
import torch
#from jarvis.arm.online.reward_model.reward_model import RewardActor
#ray.init()
ray.init(address="auto", ignore_reinit_error=True, namespace="online")
logger = logging.getLogger("Main")

torch.backends.cudnn.benchmark = False # type: ignore

@ray.remote
class TrainingSession:
    def __init__(self, logger_config: DictConfig, hyperparams: DictConfig):
        self.session_id = str(uuid.uuid4())
        hyperparams_dict = OmegaConf.to_container(hyperparams, resolve=True)
        wandb.init(config=hyperparams_dict, **logger_config) # type: ignore
    
    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)
    
    def define_metric(self, *args, **kwargs):
        wandb.define_metric(*args, **kwargs)
    
    def log_video(self, data: dict, video_key: str, fps: int):
        data[video_key] = wandb.Video(data[video_key], fps=fps, format="mp4")
        wandb.log(data)

    def get_session_id(self):
        return self.session_id

@hydra.main(config_path="configs", config_name="ppo_steve1_mineblock_final", version_base='1.1')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    rollout_manager = get_rollout_manager()
    rollout_manager_kwargs = dict(
        model_spec=cfg.model,
        env_spec=cfg.environment,
        discount=cfg.train_config.discount,
        use_normalized_vf=cfg.train_config.use_normalized_vf,
        **cfg.rollout_config
    )


    if rollout_manager is not None:
        if (ray.get(rollout_manager.get_saved_config.remote()) != rollout_manager_kwargs):
            logger.warning("Rollout manager config changed, killing and restarting rollout manager")
            ray.kill(rollout_manager)
            rollout_manager = None
        else:
            logger.info("Reusing existing rollout manager")

    if rollout_manager is None:
        if cfg.detach_rollout_manager:
            rollout_manager = RolloutManager.options(name="rollout_manager", lifetime="detached").remote(**rollout_manager_kwargs) # type: ignore
        else :
            rollout_manager = RolloutManwager.options(name="rollout_manager").remote(**rollout_manager_kwargs) # type: ignore
        # breakpoint()
        ray.get(rollout_manager.start.remote())

    training_session = None
    try:
        training_session = ray.get_actor("training_session")
    except ValueError:
        pass
    if training_session is not None:
        logger.error("Trainer already running!")
        return
    
    training_session = TrainingSession.options(name="training_session").remote(hyperparams=cfg, logger_config=cfg.logger_config) # type: ignore
    ray.get(training_session.get_session_id.remote()) # Assure that the session is created before the trainer
    ray.get(rollout_manager.update_training_session.remote())
    print("Making trainer")
    trainer = registry.get_trainer_class(cfg.trainer_name)(
        rollout_manager=rollout_manager,
        model_spec=cfg.model,
        env_spec=cfg.environment,
        **cfg.train_config
    )
    if cfg.get("enable_reward_model", False):
        from jarvis.arm.online.reward_model.reward_model import RewardActor
        actor0 = RewardActor.options(name="reward_actor_0", lifetime = "detached").remote()
        actor1 = RewardActor.options(name="reward_actor_1", lifetime = "detached").remote()
    
    trainer.fit()

if __name__ == "__main__":
    main()