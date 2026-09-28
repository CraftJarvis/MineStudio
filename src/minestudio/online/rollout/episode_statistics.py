import minestudio.online.utils.train.wandb_logger as wandb_logger
import ray
import torchmetrics
from typing import Dict, Any, Optional, List
import numpy as np
from collections import deque
import logging
from rich import print
logger = logging.getLogger("ray")
@ray.remote
class EpisodeStatistics:
    """
    A Ray actor class for collecting and logging episode statistics.

    :param discount: The discount factor for calculating discounted rewards.
    """
    def __init__(self, discount: float):
        self.discount = discount
        self.episode_info = {}
        #Maintain separate metrics for each task
        self.sum_rewards_metrics = {}#torchmetrics.MeanMetric()
        self.discounted_rewards_metrics = {}#torchmetrics.MeanMetric()
        self.episode_lengths_metrics = {}#torchmetrics.MeanMetric()
        self.episode_count_metrics = {}
        self.acc_episode_count = 0
        self.record_requests = deque()

    def update_training_session(self):
        """
        Defines metrics for wandb logging.
        """
        wandb_logger.define_metric("episode_statistics/step")
        wandb_logger.define_metric("episode_statistics/*", step_metric="episode_statistics/step")

    def log_statistics(self, step: int, record_next_episode: bool):
        """
        Logs the computed statistics to wandb and resets metrics.

        :param step: The current training step.
        :param record_next_episode: A boolean indicating whether to record the next episode.
        """
        if self.acc_episode_count == 0:
            pass
        else:
            sum_train_reward = 0
            num_train_tasks = 0
            sum_test_reward = 0
            num_test_tasks = 0
            sum_discounted_reward = 0
            sum_episode_length = 0
            num_valid_episode_length = 0  # Track valid episode length count

            # [Change log] Move the logging of step to the beginning by zhancun
            wandb_logger.log({
                "episode_statistics/step": step, 
            })  

            for task in self.sum_rewards_metrics.keys():
                mean_sum_reward = self.sum_rewards_metrics[task].compute()
                mean_discounted_reward = self.discounted_rewards_metrics[task].compute()
                mean_episode_length = self.episode_lengths_metrics[task].compute()
                episode_count = self.episode_count_metrics[task].compute()

                # Log individual task metrics
                if not np.isnan(mean_sum_reward):
                    wandb_logger.log({
                        f"episode_statistics/{task}/sum_reward": mean_sum_reward,
                        f"episode_statistics/{task}/discounted_reward": mean_discounted_reward,
                        f"episode_statistics/{task}/episode_length": mean_episode_length,
                        f"episode_statistics/{task}/episode_count": episode_count,
                        f"episode_statistics/{task}/frequency": episode_count / self.acc_episode_count,
                    })
                    print(f"Task {task} - Sum Reward: {mean_sum_reward}, Discounted Reward: {mean_discounted_reward}, Episode Length: {mean_episode_length}")

                self.sum_rewards_metrics[task].reset()
                self.discounted_rewards_metrics[task].reset()
                self.episode_lengths_metrics[task].reset()
                self.episode_count_metrics[task].reset()
                
                if not np.isnan(mean_sum_reward) and "4train" in task:
                    sum_train_reward += mean_sum_reward
                    sum_discounted_reward += mean_discounted_reward
                    num_train_tasks += 1
                if not np.isnan(mean_sum_reward) and "4test" in task:
                    sum_test_reward += mean_sum_reward
                    num_test_tasks += 1
                
                # Only add episode length if it's not NaN
                if not np.isnan(mean_episode_length) and not np.isnan(mean_sum_reward):
                    sum_episode_length += mean_episode_length
                    num_valid_episode_length += 1

            self.episode_info = {
                "steps": step,
                "episode_count": self.acc_episode_count,
                "mean_sum_reward": sum_train_reward / num_train_tasks if num_train_tasks > 0 else 0,
                "mean_discounted_reward": sum_discounted_reward / num_train_tasks if num_train_tasks > 0 else 0,
                "mean_episode_length": sum_episode_length / num_valid_episode_length if num_valid_episode_length > 0 else 0
            }
            wandb_logger.log({
                "episode_statistics/steps": step,
                "episode_statistics/episode_count": self.acc_episode_count,
                "episode_statistics/mean_sum_reward": sum_train_reward / num_train_tasks if num_train_tasks > 0 else 0,
                "episode_statistics/mean_test_sum_reward": sum_test_reward / num_test_tasks if num_test_tasks > 0 else 0,
                "episode_statistics/mean_discounted_reward": sum_discounted_reward / num_train_tasks if num_train_tasks > 0 else 0,
                "episode_statistics/mean_episode_length": sum_episode_length / num_valid_episode_length if num_valid_episode_length > 0 else 0
            })

            self.acc_episode_count = 0
            
        print("received_episode_statistics:"+str(step)+str(record_next_episode))
        if record_next_episode:
            if len(self.record_requests) > 0:
                print("There are still unprocessed record requests.")
                logger.warning("There are still unprocessed record requests.")
            else:
                #! shaofei modify
                # self.record_requests.append(step)
                for i in range(4):
                    self.record_requests.append(step)
                print("append_record_requests:"+str(self.record_requests))
    
    def report_episode(self, rewards: np.ndarray, its_specfg: str="", additional_des: str="4train") -> Optional[int]:
        """
        Reports the rewards for a completed episode and updates metrics.

        :param rewards: A NumPy array of rewards for the episode.
        :param its_specfg: A string specifying the task configuration.
        :param additional_des: Additional description for the task (e.g., "4train" or "4test").
        :returns: A tuple containing the step number and episode information if a video record is requested, otherwise None and episode information.
        """
        its_specfg = its_specfg + additional_des
        if its_specfg not in self.sum_rewards_metrics:
            self.sum_rewards_metrics[its_specfg] = torchmetrics.MeanMetric()
            self.discounted_rewards_metrics[its_specfg] = torchmetrics.MeanMetric()
            self.episode_lengths_metrics[its_specfg] = torchmetrics.MeanMetric()
            self.episode_count_metrics[its_specfg] = torchmetrics.SumMetric()
        sum_reward = rewards.sum()

        discounted_reward = ((self.discount ** np.arange(len(rewards))) * rewards).sum()
        episode_length = len(rewards)
        self.sum_rewards_metrics[its_specfg].update(sum_reward)
        self.discounted_rewards_metrics[its_specfg].update(discounted_reward)
        self.episode_lengths_metrics[its_specfg].update(episode_length)
        self.episode_count_metrics[its_specfg].update(1)
        self.acc_episode_count += 1
        if len(self.record_requests) > 0:
            print("episode, cord_requests>0:" + str(self.record_requests))
            step = self.record_requests.popleft()
            return step, self.episode_info
        else:
            return None, self.episode_info