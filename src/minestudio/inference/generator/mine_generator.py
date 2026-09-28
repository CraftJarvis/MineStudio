'''
Date: 2024-11-25 08:35:59
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-02 11:59:51
FilePath: /MineStudio/minestudio/inference/generator/mine_generator.py
'''
import os
import ray

from typing import Callable, Optional, List, Dict, Tuple, Literal, Generator

from minestudio.inference.generator.base_generator import EpisodeGenerator, AgentInterface
from minestudio.utils import get_compute_device

class Worker:
    """
    A worker class for generating episodes.

    This class handles the interaction between an environment and an agent
    to generate a specified number of episodes, each with a maximum number of steps.
    It saves the generated data (images, actions, infos) to files.

    :param env_generator: A function that generates an environment instance.
    :type env_generator: Callable
    :param agent_generator: A function that generates an agent instance.
    :type agent_generator: Callable
    :param num_max_steps: The maximum number of steps per episode.
    :type num_max_steps: int
    :param num_episodes: The number of episodes to generate.
    :type num_episodes: int
    :param tmpdir: The temporary directory to save episode data. Defaults to None.
    :type tmpdir: Optional[str]
    :param image_media: The format to save images ("h264" or "jpeg"). Defaults to "h264".
    :type image_media: Literal["h264", "jpeg"]
    :param unused_kwargs: Additional unused keyword arguments.
    """

    def __init__(
        self, 
        env_generator: Callable, 
        agent_generator: Callable, 
        num_max_steps: int, 
        num_episodes: int, 
        tmpdir: Optional[str] = None, 
        image_media: Literal["h264", "jpeg"] = "h264",
        **unused_kwargs,
    ):
        """
        Initializes the Worker.

        :param env_generator: A function that generates an environment instance.
        :type env_generator: Callable
        :param agent_generator: A function that generates an agent instance.
        :type agent_generator: Callable
        :param num_max_steps: The maximum number of steps per episode.
        :type num_max_steps: int
        :param num_episodes: The number of episodes to generate.
        :type num_episodes: int
        :param tmpdir: The temporary directory to save episode data. Defaults to None.
        :type tmpdir: Optional[str]
        :param image_media: The format to save images ("h264" or "jpeg"). Defaults to "h264".
        :type image_media: Literal["h264", "jpeg"]
        :param unused_kwargs: Additional unused keyword arguments.
        """
        self.num_max_steps = num_max_steps
        self.num_episodes = num_episodes
        self.env = env_generator()
        self.agent = agent_generator().to(get_compute_device())
        self.agent.eval()
        self.image_media = image_media
        self.tmpdir = tmpdir

        self.generator = self._run()
        os.makedirs(self.tmpdir, exist_ok=True)

    def append_image_and_info(self, info: Dict, images: List, infos: List):
        """
        Appends image and info data to the respective lists.

        The 'pov' (point of view) image is extracted from the info dictionary.
        The info dictionary is cleaned to ensure all values are of basic dict type if they have a 'values' attribute.

        :param info: The info dictionary from the environment.
        :type info: Dict
        :param images: The list to append the image to.
        :type images: List
        :param infos: The list to append the cleaned info to.
        :type infos: List
        """
        info = info.copy()
        image = info.pop("pov")
        for key, val in info.items(): # use clean dict type
            if hasattr(info[key], 'values'):
                info[key] = dict(info[key])
        images.append(image)
        infos.append(info)

    def save_to_file(self, images: List, actions: List, infos: List):
        """
        Saves the episode data (images, actions, infos) to files.

        Generates a unique episode ID and saves infos and actions as pickle files.
        Saves images as an H264 video or a series of JPEG images based on `self.image_media`.

        :param images: A list of images (frames) from the episode.
        :type images: List
        :param actions: A list of actions taken during the episode.
        :type actions: List
        :param infos: A list of info dictionaries from the episode.
        :type infos: List
        :returns: A dictionary containing the paths to the saved files.
        :rtype: Dict
        :raises ValueError: If `self.image_media` is not "h264" or "jpeg".
        """
        import av, pickle, uuid
        from PIL import Image
        episode_id = str(uuid.uuid4())
        episode = {}
        episode["info_path"] = f"{self.tmpdir}/info_{episode_id}.pkl"
        with open(episode["info_path"], "wb") as f:
            pickle.dump(infos, f)
        episode["action_path"] = f"{self.tmpdir}/action_{episode_id}.pkl"
        with open(episode["action_path"], "wb") as f:
            pickle.dump(actions, f)
        if self.image_media == "h264":
            episode["video_path"] = f"{self.tmpdir}/video_{episode_id}.mp4"
            with av.open(episode["video_path"], mode="w", format='mp4') as container:
                stream = container.add_stream("h264", rate=30)
                stream.width = images[0].shape[1]
                stream.height = images[0].shape[0]
                for image in images:
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        elif self.image_media == "jpeg":
            episode["base_image_path"] = f"{self.tmpdir}/images_{episode_id}"
            os.makedirs(episode["base_image_path"], exist_ok=True)
            for i, image in enumerate(images):
                image = Image.fromarray(image)
                image.save(f"{episode['base_image_path']}/{i}.jpeg")
        else:
            raise ValueError(f"Invalid image_media: {self.image_media}")
        return episode

    def _run(self) -> Generator[Dict, None, None]:
        """
        Runs the episode generation loop.

        Iterates for `self.num_episodes`. In each episode, it resets the environment,
        then runs for `self.num_max_steps`, collecting actions, observations, and infos.
        After each episode, it saves the data and yields the episode file paths.

        :returns: A generator that yields a dictionary of file paths for each episode.
        :rtype: Generator[Dict, None, None]
        """
        for eps_id in range(self.num_episodes):
            memory = None
            actions = []
            images = []
            infos = []
            obs, info = self.env.reset()
            self.append_image_and_info(info, images, infos)
            for step in range(self.num_max_steps):
                action, memory = self.agent.get_action(obs, memory, input_shape='*')
                actions.append(action)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.append_image_and_info(info, images, infos)
            yield self.save_to_file(images, actions, infos)
        self.env.close()

    def get_next(self):
        """
        Gets the next generated episode.

        :returns: The next episode data (dictionary of file paths), or None if generation is complete.
        :rtype: Optional[Dict]
        :raises ValueError: If the generator is not initialized.
        """
        if self.generator is None:
            raise ValueError("Generator is not initialized. Call init_generator first.")
        try:
            return next(self.generator)
        except StopIteration:
            return None

class MineGenerator(EpisodeGenerator):
    """
    A generator for Minecraft episodes using multiple parallel workers.

    This class manages multiple `Worker` instances (potentially distributed with Ray)
    to generate Minecraft episodes in parallel.

    :param num_workers: The number of parallel workers to use. Defaults to 1.
    :type num_workers: int
    :param num_gpus: The number of GPUs to assign to each Ray worker. Defaults to 0.5.
    :type num_gpus: float
    :param max_restarts: The maximum number of times a Ray worker can be restarted if it fails. Defaults to 3.
    :type max_restarts: int
    :param worker_kwargs: Keyword arguments to pass to the `Worker` constructor.
    """

    def __init__(
        self, 
        num_workers: int = 1,
        num_gpus: float = 0.5, 
        max_restarts: int = 3,
        **worker_kwargs, 
    ):
        """
        Initializes the MineGenerator.

        Creates `num_workers` remote Ray actors of the `Worker` class.

        :param num_workers: The number of parallel workers to use. Defaults to 1.
        :type num_workers: int
        :param num_gpus: The number of GPUs to assign to each Ray worker. Defaults to 0.5.
        :type num_gpus: float
        :param max_restarts: The maximum number of times a Ray worker can be restarted if it fails. Defaults to 3.
        :type max_restarts: int
        :param worker_kwargs: Keyword arguments to pass to the `Worker` constructor.
        """
        super().__init__()
        self.num_workers = num_workers
        self.workers = []
        for worker_id in range(num_workers):
            self.workers.append(
                ray.remote(
                    num_gpus=num_gpus, 
                    max_restarts=max_restarts,
                )(Worker).remote(**worker_kwargs)
            )

    def generate(self) -> Generator[Dict, None, None]:
        """
        Generates episodes in parallel using the workers.

        Uses `ray.wait` to get completed episodes from the workers.
        When a worker finishes an episode, it yields the episode data
        and assigns the worker to generate another episode.

        :returns: A generator that yields dictionaries of file paths for each episode.
        :rtype: Generator[Dict, None, None]
        """
        pools = {worker.get_next.remote(): worker for worker in self.workers}
        while pools:
            done, _ = ray.wait(list(pools.keys()))
            for task in done:
                worker = pools.pop(task)
                episode = ray.get(task)
                if episode is not None:
                    yield episode
                    pools[worker.get_next.remote()] = worker
