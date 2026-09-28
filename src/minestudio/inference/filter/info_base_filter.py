'''
Date: 2024-11-25 12:39:01
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 08:19:00
FilePath: /MineStudio/minestudio/inference/filter/info_base_filter.py
'''
import re
import pickle
from minestudio.inference.filter.base_filter import EpisodeFilter

class InfoBaseFilter(EpisodeFilter):
    """
    A filter that filters episodes based on information extracted from a specified key in the episode's info.

    It uses a regular expression to match events in the info and counts the occurrences.
    If the total count meets a specified number, the episode is labeled.

    :param key: The key in the episode info to extract data from.
    :type key: str
    :param regex: The regular expression to match events.
    :type regex: str
    :param num: The minimum number of matched events required for an episode to pass the filter.
    :type num: int
    :param label: The label to assign to the episode if it passes the filter. Defaults to "status".
    :type label: str
    """
    
    def __init__(self, key: str, regex: str, num: int, label: str = "status"):
        """
        Initializes the InfoBaseFilter.

        :param key: The key in the episode info to extract data from.
        :type key: str
        :param regex: The regular expression to match events.
        :type regex: str
        :param num: The minimum number of matched events required for an episode to pass the filter.
        :type num: int
        :param label: The label to assign to the episode if it passes the filter. Defaults to "status".
        :type label: str
        """
        self.key = key
        self.regex = regex
        self.num = num
        self.label = label
    
    def filter(self, episode_generator):
        """
        Filters episodes based on the specified criteria.

        For each episode, it loads the info, extracts data based on the key,
        matches events using the regex, and counts the occurrences.
        If the total count is greater than or equal to num, the episode is labeled.

        :param episode_generator: A generator that yields episodes.
        :type episode_generator: Generator
        :returns: A generator that yields filtered episodes.
        :rtype: Generator
        """
        for episode in episode_generator:
            info = pickle.loads(open(episode["info_path"], "rb").read())
            total = 0
            last_info = info[-1][self.key]
            for event in last_info:
                if re.match(self.regex, event):
                    total += last_info.get(event, 0)
            if total >= self.num:
                episode[self.label] = "yes"
            yield episode
