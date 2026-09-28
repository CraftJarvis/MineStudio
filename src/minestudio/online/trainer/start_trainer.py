import ray
from minestudio.online.utils.rollout import get_rollout_manager
import minestudio.online.trainer
from minestudio.online.utils.train.training_session import TrainingSession

def start_trainer(policy_generator, env_generator, online_cfg, whole_config):
    """Starts the training process.

    This function initializes and starts a training session, creating a trainer
    instance based on the provided configuration.

    :param policy_generator: A function to generate the policy model.
    :param env_generator: A function to generate the environment.
    :param online_cfg: Online training configuration.
    :param whole_config: The entire configuration as a string.
    """
    training_session = None
    try:
        training_session = ray.get_actor("training_session")
    except ValueError:
        pass
    if training_session is not None:
        print("Trainer already running!")
        return

    training_session = TrainingSession.options(name="training_session").remote(hyperparams=online_cfg, logger_config=online_cfg.logger_config) # type: ignore
    ray.get(training_session.get_session_id.remote()) # Assure that the session is created before the trainer

    print("Making trainer")
    trainer_class_name = online_cfg.trainer_name  # 字符串，如 "DQNTrainer"
    trainer_class = getattr(minestudio.online.trainer, trainer_class_name, None)
    rollout_manager = get_rollout_manager()
    trainer = trainer_class(
        rollout_manager=rollout_manager,
        policy_generator=policy_generator,
        env_generator=env_generator,
        **online_cfg.train_config,
        whole_config = whole_config
    )
    trainer.fit()