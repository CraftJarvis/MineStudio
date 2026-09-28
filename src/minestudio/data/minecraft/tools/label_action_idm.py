'''
Date: 2024-11-10 12:28:23
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 12:30:32
FilePath: /MineStudio/minestudio/data/minecraft/tools/label_action_idm.py
'''

import os
import time
import argparse
import random
import pickle
import redis
import torch
import numpy as np
import torch.multiprocessing as mp

import cv2

from rich import print
from rich.console import Console
from pathlib import Path
from itertools import chain

import sys
sys.path.append('./idm_lib')

from idm_lib.agent import ENV_KWARGS
from idm_lib.inverse_dynamics_model import IDMAgent

console = Console()

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}


CAMERA_SCALER = 360.0 / 2400.0
VIDEO_ID_LENGTH = 11

class LabelWorker(mp.Process):
    
    def __init__(self, pipe: mp.Pipe, device: int, model: str, weights: str, save_dir: str):
        super(LabelWorker, self).__init__()
        self.pipe = pipe
        self.device = device
        self.model_path = model
        self.weights_path = weights
        self.save_dir = save_dir


    def init_IDM_model(self):
        agent_parameters = pickle.load(open(self.model_path, "rb"))
        net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, device=self.device)
        self.agent.load_weights(self.weights_path)


    def run(self):
        
        self.init_IDM_model()
        console.log(f"<worker pid: {os.getpid()}> start running at cuda device {self.device}")
        
        while True:
            
            self._send_message("free", None)
            
            command, args = self._recv_message()
            
            if command == 'process':
                task = args[0]
                flag = self.process(task)
                console.log(f"<worker {self.device}> finished task {task}, flag: {flag}.")
                self._send_message("finish", (task, flag))
            
            if command == 'close':
                console.log(f"<worker {self.device}> done.")
                return 

    def process(self, task):
        vid, video_path = task
        required_resolution = ENV_KWARGS["resolution"]
        cap = cv2.VideoCapture(str(video_path.absolute()))
        cv_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cv_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if cv_width != required_resolution[0] or cv_height != required_resolution[1]:
            return False
        
        # action_results = []
        # while True:
        #     torch.cuda.empty_cache()
        #     eof = False
        #     frames = []
        #     for _ in range(128):
        #         ret, frame = cap.read()
        #         if not ret:
        #             eof = True
        #             break
        #         rgb_frame = frame[..., ::-1]
        #         frames.append(rgb_frame)
        #     # corner case
        #     if len(frames) == 0:
        #         break
        #     frames = np.stack(frames)
        #     predicted_actions = self.agent.predict_actions(frames)
        #     action_results.append(predicted_actions)
        #     if eof:
        #         break
        
        action_results = []
        frames = []
        slice_fn = lambda dic, l, r: {k: v[:,l:r] for k, v in dic.items()}
        while True:
            torch.cuda.empty_cache()
            while len(frames) < 128:
                ret, frame = cap.read()
                if not ret: break
                rgb_frame = frame[..., ::-1]
                frames.append(rgb_frame)
            
            input = np.stack(frames)
            predicted_actions = self.agent.predict_actions(input)

            if len(action_results) == 0:
                foo = slice_fn(predicted_actions, 0, 32)
                action_results.append(foo)

            if len(frames) < 128:
                foo = slice_fn(predicted_actions, 32, None)
                action_results.append(foo)
                break
            elif len(frames) == 128:
                foo = slice_fn(predicted_actions, 32, 96)
                action_results.append(foo)
            
            frames = frames[64:]
        cap.release()
        
        res = {}
        for dic in action_results:
            for key, val in dic.items():
                if key not in res:
                    res[key] = val[0]
                else: 
                    res[key] = np.concatenate((res[key], val[0]), axis=0)
        
        assert int(total_frames) == res['attack'].shape[0]
        
        p = Path(self.save_dir)/f"{vid}.pkl"
        with p.open("wb") as f:
            pickle.dump(res, f)
        return True


    def _send_message(self, command, args):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args


class Manager:
    
    def __init__(self, args):
        self.nb_worker = args.nb_worker
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        visible_devices = visible_devices.split(',')
        assert len(visible_devices) >= self.nb_worker, "violate rule: param [nb_worker] <= [visible_devices]. "
        
        console.log(f"enabled devices: {visible_devices}")
        self.source_redis_key = args.redis_key
        self.visited_redis_key = "visited:" + self.source_redis_key
        self.redis_cli = redis.StrictRedis(host=args.redis_host, port=args.redis_port, db=0)
        task_list = self.redis_cli.smembers(self.source_redis_key)
        task_list = [vid.decode("utf-8") for vid in task_list]
        console.log(f"Candidate video number: {len(task_list)}")
        # done_tasks = chain(self.redis_cli.smembers('successful_labeled_video_id'), self.redis_cli.smembers('failed_labeled_video_id'))
        done_tasks = {x.decode("utf-8") for x in self.redis_cli.smembers(self.visited_redis_key)}
        console.log(f"Action labeled video number: {len(done_tasks)}")
        # task_list = [x for x in task_list if x not in done_tasks]
        # console.log(f"To be labeled video number: {len(task_list)}")
        
        self.task_list = []
        for video in Path(args.video_dir).glob('*.mp4'):
            # assert video.name[11] == '_', "invalid video name format"
            vid = video.name[:VIDEO_ID_LENGTH]
            if len(task_list) > 0:
                if vid not in task_list:
                    continue
            if vid in done_tasks:
                continue
            self.task_list.append((vid, video))
        console.log(f"task list: {len(self.task_list)}")
        
        self._env_workers = []
        self._pipes = []
        for i in range(self.nb_worker):
            parent_pipe, child_pipe = mp.Pipe()
            self._pipes.append(parent_pipe)
            worker = LabelWorker(
                pipe=child_pipe, 
                device=i,
                model=args.model,
                weights=args.weights,
                save_dir=args.save_dir,
            )
            self._env_workers.append(worker)
            
        for worker in self._env_workers:
            worker.start()
    
    
    def run(self):
        
        opening = self.nb_worker
        while opening > 0:
            
            for worker_idx in range(self.nb_worker):
                command, args = self._recv_message_nonblocking(worker_idx)
                
                if command is None:
                    continue
                
                if command == "free":
                    
                    task = None
                    if len(self.task_list) > 0:
                        task = self.task_list.pop()
                    # while len(self.task_list) > 0:
                    #     task = self.task_list.pop()
                    #     if self.redis_cli.sismember("successful_labeled_video_id", task[0]) or self.redis_cli.sismember("failed_labeled_video_id", task[0]):
                    #         continue
                    #     else:
                    #         break

                    if task is not None:
                        console.log(f"Remain: {len(self.task_list)} | Fetch task: {task} and Assign to worker {worker_idx}.")
                        self._send_message(worker_idx, "process", (task, ))
                    else:
                        opening -= 1
                        self._send_message(worker_idx, "close", None)
                
                elif command == 'finish':
                    vid, _ = args[0]
                    done_flag = args[1]
                    if done_flag:
                        self.redis_cli.sadd(self.visited_redis_key, vid)
                
        console.log("All tasks have been processed. ")

    def _broadcast_message(self, command, args = None):
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, command, args = args)

    def _send_message(self, worker_idx, command, args = None):
        self._pipes[worker_idx].send((command, args))

    def _recv_message_nonblocking(self, worker_idx):
        if not self._pipes[worker_idx].poll():
            return None, None

        command, args = self._pipes[worker_idx].recv()
        return command, args

if __name__ == '__main__':
    
    mp.set_start_method('forkserver')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-host", type=str, default='127.0.0.1')
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-key", type=str, default="none", 
                        help="redis key that specifies videos to be processed")
    parser.add_argument("--nb-worker", type=int, default=1, 
                        help="number of paralle workers")
    parser.add_argument("--model", type=str, required=True,
                        help="the IDM model path")
    parser.add_argument("--weights", type=str, required=True,
                        help="the IDM weights path")
    parser.add_argument("--video-dir", type=str, required=True, 
                        help="the directory containing all the videos")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="the directory saving the predicted actions")


    args = parser.parse_args()
    p = Path(args.save_dir)
    if not p.is_dir():
        p.mkdir()
    manager = Manager(args)
    manager.run()
    