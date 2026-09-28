'''
Date: 2024-11-25 07:36:18
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 12:06:21
FilePath: /MineStudio/minestudio/inference/generator/base_generator.py
'''
from abc import abstractmethod
from typing import List, Dict, Any, Tuple, Generator

class EpisodeGenerator:
    """
    Base class for episode generators.

    Generators are responsible for creating sequences of experiences (episodes).
    """
    
    def __init__(self):
        """
        Initializes the EpisodeGenerator.
        """
        pass

    @abstractmethod
    def generate(self) -> Generator:
        """
        Generates episodes.

        This method must be implemented by subclasses to define how episodes are generated.

        :returns: A generator that yields episodes.
        :rtype: Generator
        """
        pass

class AgentInterface:
    """
    Interface for an agent that can interact with an environment.
    """
    
    @abstractmethod
    def get_action(self, input: Dict, state: Any, **kwargs) -> Tuple[Any, Any]:
        """
        Gets an action from the agent.

        This method must be implemented by subclasses to define how the agent selects an action
        based on the input and current state.

        :param input: The input to the agent (e.g., observations from the environment).
        :type input: Dict
        :param state: The current state of the agent.
        :type state: Any
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing the action selected by the agent and the updated state.
        :rtype: Tuple[Any, Any]
        """
        pass