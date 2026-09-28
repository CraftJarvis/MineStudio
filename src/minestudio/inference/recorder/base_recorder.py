'''
Date: 2024-11-25 07:35:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 12:55:03
FilePath: /MineStudio/minestudio/inference/recorder/base_recorder.py
'''
from abc import abstractmethod
from typing import List, Dict, Union, Generator

class EpisodeRecorder:
    """
    Base class for episode recorders.

    Recorders are used to process and summarize a collection of episodes.
    """

    def __init__(self):
        """
        Initializes the EpisodeRecorder.
        """
        pass

    def record(self, episode_generator: Generator) -> Union[Dict, str]:
        """
        Records and summarizes episodes from an episode generator.

        This method iterates through the episodes, counts the total number of episodes
        and the number of episodes marked with "status" == "yes".
        It returns a dictionary containing these counts and the 'yes_rate'.

        :param episode_generator: A generator that yields episodes.
        :type episode_generator: Generator
        :returns: A dictionary with a summary of the episodes, including 'num_yes', 'num_episodes', and 'yes_rate'.
        :rtype: Union[Dict, str]
        """
        num_yes = 0
        num_episodes = 0
        for episode in episode_generator:
            num_episodes += 1
            if episode.get("status") == "yes":
                num_yes += 1
        return {
            "num_yes": num_yes,
            "num_episodes": num_episodes,
            "yes_rate": f"{num_yes / num_episodes * 100:.2f}%",
        }