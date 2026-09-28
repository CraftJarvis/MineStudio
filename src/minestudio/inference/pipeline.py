'''
Date: 2024-11-25 07:29:21
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 12:40:22
FilePath: /MineStudio/minestudio/inference/pipeline.py
'''

import ray
from typing import Union, List, Optional
from minestudio.inference.generator.base_generator import EpisodeGenerator
from minestudio.inference.filter.base_filter import EpisodeFilter
from minestudio.inference.recorder.base_recorder import EpisodeRecorder

class EpisodePipeline:
    """
    A pipeline for generating, filtering, and recording episodes.

    :param episode_generator: An instance of EpisodeGenerator.
    :type episode_generator: EpisodeGenerator
    :param episode_filter: An instance of EpisodeFilter or a list of EpisodeFilter instances. Defaults to None.
    :type episode_filter: Optional[Union[EpisodeFilter, List[EpisodeFilter]]]
    :param episode_recorder: An instance of EpisodeRecorder. Defaults to None.
    :type episode_recorder: Optional[EpisodeRecorder]
    """

    def __init__(
        self, 
        episode_generator: EpisodeGenerator,
        episode_filter: Optional[Union[EpisodeFilter, List[EpisodeFilter]]] = None,
        episode_recorder: Optional[EpisodeRecorder] = None,
    ):
        """
        Initializes the EpisodePipeline.

        :param episode_generator: An instance of EpisodeGenerator.
        :type episode_generator: EpisodeGenerator
        :param episode_filter: An instance of EpisodeFilter or a list of EpisodeFilter instances. Defaults to None.
        :type episode_filter: Optional[Union[EpisodeFilter, List[EpisodeFilter]]]
        :param episode_recorder: An instance of EpisodeRecorder. Defaults to None.
        :type episode_recorder: Optional[EpisodeRecorder]
        """
        if episode_filter is None:
            episode_filter = EpisodeFilter()
        if episode_recorder is None:
            episode_recorder = EpisodeRecorder()
        if not isinstance(episode_filter, List):
            episode_filter = [episode_filter]

        self.episode_filter = episode_filter
        self.episode_generator = episode_generator
        self.episode_recorder = episode_recorder 

    def run(self):
        """
        Runs the episode pipeline.

        The pipeline generates an episode, filters it, and then records it.

        :returns: A summary of the recorded episode.
        :rtype: Any
        """
        _generator = self.episode_generator.generate()
        for episode_filter in self.episode_filter:
            _generator = episode_filter.filter(_generator)
        summary = self.episode_recorder.record(_generator)
        return summary
