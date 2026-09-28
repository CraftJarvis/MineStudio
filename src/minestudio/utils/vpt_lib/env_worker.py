from ctypes import resize
import multiprocessing as mp
import numpy as np
import time
import cv2
import lmdb
import uuid
import os
import json
import pickle
from rich import print
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from minestudio.utils.vpt_lib.serialize import serialize_object


def resize_image(img, target_resolution = (227, 128)):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img

# ignores = []


# ignores = ["sheep", "cow", "pig", "dirt", "stone", "sand", "Oak Wood", "Oak Sapling", "Birch Wood", "Birch Sapling", "Spruce Wood", "Spruce Sapling",
#             "Dandelion", "Poppy", "Blue Orchid", "Allium", "Azure Bluet", "Red Tulip", "Orange Tulip", "White Tulip", "Pink Tulip", "Oxeye Daisy", "Brown Mushroom",
#             "Pumpkin", "Sunflower", "Lilac", "Double Tallgrass", "Large Fern", "Rose Bush", "Peony", "Wheat seeds", "Sugar Canes",]

ignores = ["pig", "cow"]


class EnvWorker(mp.Process):
    def __init__(self, pipe, env_generator, verbose = False, recording_path = None, voxels = False, always_record = False):
        super(EnvWorker, self).__init__()

        self.pipe = pipe
        self.env_generator = env_generator
        self.verbose = verbose
        self.recording_path = recording_path
        self.record = recording_path is not None
        self.always_record = always_record
        self.voxels = voxels
        self.worker_id = None if recording_path is None else int(recording_path.split("_")[-1])

    def init_process(self):
        self._env = self.env_generator()

        if self.record:
            self._batch_builder = SampleBatchBuilder()
            self._writer = JsonWriter(self.recording_path)

    def run(self):
        self.init_process()

        total_wait_time = 0.0
        wait_count = 0
        total_occupied_time = 0.0
        occupacy_count = 0

        prev_obs = None
        global_eps_id = 0
        time_step = 0
        cum_reward = 0.0
        successful_eps = []

        total_num_eps = 0

        while True:

            wait_start = time.time()

            command, args = self._recv_message()

            wait_end = time.time()
            total_wait_time += wait_end - wait_start
            wait_count += 1
            if self.verbose:
                print("[worker {}] aveg idle time: {:.2f}ms; aveg working time: {:.2f}ms".format(
                    mp.current_process().name, total_wait_time / wait_count * 1000, total_occupied_time / (occupacy_count + 1e-8) * 1000))

            if command == "reset":
                obs = self._env.reset()
                obs['rgb'] = resize_image(obs['rgb'], target_resolution = (160, 120))
                self._send_message("send_obs", obs)

                if self.record:
                    prev_obs = obs
                    time_step = 0
                    global_eps_id += 1
                    cum_reward = 0.0

            elif command == "step":
                occupy_start = time.time()

                action = args
                obs, reward, done, info = self._env.step(action)
                obs['rgb'] = resize_image(obs['rgb'], target_resolution = (160, 120))

                simplified_info = dict()
                simplified_info["accomplishments"] = info["accomplishments"]

                self._send_message("send_obs_plus", (obs, reward, done, simplified_info))

                occupy_end = time.time()
                total_occupied_time += occupy_end - occupy_start
                occupacy_count += 1

                if self.record:
                    if self.voxels:
                        voxels = info["voxels"]["block_name"]
                        voxel_set = set()
                        for i in range(voxels.shape[0]):
                            for j in range(voxels.shape[1]):
                                for k in range(voxels.shape[2]):
                                    voxel_set.add(voxels[i][j][k])
                        voxel_list = list(voxel_set)
                    else:
                        voxel_list = []
                        
                    if len(info["accomplishments"]) > 0:
                        print(info["accomplishments"])
                        
                    self._batch_builder.add_values(
                        eps_id = global_eps_id,
                        t = time_step,
                        agent_index = 0,
                        obs = prev_obs,
                        action = action,
                        reward = reward,
                        done = done,
                        accomplishments = info["accomplishments"],
                        voxel_list = voxel_list,
                        info = info,
                    )
                    # import json
                    # with open("/home/caishaofei/Desktop/info.json", "wb") as f:
                    #     json.dump(f, info)
                    # exit()
                    time_step += 1
                    prev_obs = obs
                    cum_reward += reward

                    if done and (self.always_record or cum_reward > 0.0) and (info['accomplishments'][-1] not in ignores):
                        successful_eps.append(global_eps_id)
                        print(f"> Worker {self.worker_id} completed 1 successful eps ({len(successful_eps)} buffered)")

                    if done and len(successful_eps) >= 5:
                        batches = self._batch_builder.build_and_reset_with_cond(successful_eps)
                        total_num_eps += len(successful_eps)
                        successful_eps.clear()
                        self._writer.write(batches)
                        print(f"> [Json] Worker {self.worker_id} dumped 5 eps ({total_num_eps} in total)")

            elif command == "kill_proc":
                return


    def _send_message(self, command, args):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args


class EnvManager():
    def __init__(self, env_generator, num_workers, verbose = False, recording_path = None, voxels = False, 
                 always_record = False, use_lmdb = False, lmdb_chunk_size = 8):
        self.env_generator = env_generator
        self.num_workers = num_workers
        self.verbose = verbose

        self.time_steps = None
        self.last_obs = None
        self.last_action = None
        self.eps_id = None
        self.cum_rewards = None
        self.use_lmdb = use_lmdb
        self.lmdb_chunk_size = lmdb_chunk_size

        self.num_used_eps = 0

        self.workers = []
        self.pipes = []

        self.worker_traj_history = None

        self.total_n_recorded_eps = 0
        self.per_task_n_recorded_eps = dict()

        if use_lmdb and recording_path is not None:
            self.lmdb_trajs_path = os.path.join(recording_path, "trajs")
            self.lmdb_indices_path = os.path.join(recording_path, "indices")
            if not os.path.exists(recording_path):
                os.mkdir(recording_path)
            if not os.path.exists(self.lmdb_trajs_path):
                os.mkdir(self.lmdb_trajs_path)
            if not os.path.exists(self.lmdb_indices_path):
                os.mkdir(self.lmdb_indices_path)

            self.trajs_lmdb_env = lmdb.open(self.lmdb_trajs_path, map_size = 1099511627776)
            self.indices_lmdb_env = lmdb.open(self.lmdb_indices_path, map_size = 1073741824)

            self.worker_traj_history = [None for _ in range(self.num_workers)]

            for i in range(self.num_workers):
                parent_pipe, child_pipe = mp.Pipe()
                self.pipes.append(parent_pipe)

                worker = EnvWorker(child_pipe, self.env_generator, verbose = verbose, recording_path = None, 
                                   voxels = voxels, always_record = False)
                self.workers.append(worker)
        else:
            for i in range(self.num_workers):
                parent_pipe, child_pipe = mp.Pipe()
                self.pipes.append(parent_pipe)

                curr_rec_path = None if recording_path is None else recording_path + f"_{i}"
                worker = EnvWorker(child_pipe, self.env_generator, verbose = verbose, recording_path = curr_rec_path, 
                                voxels = voxels, always_record = always_record)
                self.workers.append(worker)

    def start(self):
        for worker in self.workers:
            worker.start()

    def reset_all_envs(self):
        self._broadcast_message("reset")

    def collect_observations(self):
        observations = []
        worker_ids = []
        for worker_idx in range(self.num_workers):
            command, args = self._recv_message_nonblocking(worker_idx)
            if command is None:
                continue

            if command == "send_obs":
                obs = args
                observations.append(obs)
                worker_ids.append(worker_idx)

                if self.worker_traj_history is not None:
                    self.worker_traj_history[worker_idx] = [[obs]]

            elif command == "send_obs_plus":
                obs, reward, done, info = args
                if not done:
                    observations.append(obs)
                    worker_ids.append(worker_idx)
                else:
                    self._send_message(worker_idx, "reset")

                if self.worker_traj_history is not None:
                    self.worker_traj_history[worker_idx][-1].extend([reward, done, info])
                    self.worker_traj_history[worker_idx].append([obs])
                    if done:
                        self._conditional_record(worker_idx)

        return observations, worker_ids

    def execute_actions(self, actions, worker_ids):
        for action, worker_idx in zip(actions, worker_ids):
            self._send_message(worker_idx, "step", action)
            if self.worker_traj_history is not None:
                self.worker_traj_history[worker_idx][-1].append(action)

    def _conditional_record(self, worker_idx):
        cum_reward = sum([item[2] for item in self.worker_traj_history[worker_idx][:-1]])
        accomplishments = set()
        for item in self.worker_traj_history[worker_idx][:-1]:
            for accomplishment in item[4]['accomplishments']:
                accomplishments.add(accomplishment)
        accomplishments = list(accomplishments)

        save_flag = True
        for x in ignores:
            if x in accomplishments:
                save_flag = False
        
        if cum_reward > 0.0 and save_flag:
            for accomplishment in accomplishments:
                if accomplishment not in self.per_task_n_recorded_eps:
                    self.per_task_n_recorded_eps[accomplishment] = 0
                self.per_task_n_recorded_eps[accomplishment] += 1

            self.worker_traj_history[worker_idx][-1].extend([None, None, None, None])

            traj_name = str(uuid.uuid1())
            chunks = []
            traj_len = len(self.worker_traj_history[worker_idx])
            for chunk_start in range(0, traj_len, self.lmdb_chunk_size):
                chunk_end = min(chunk_start + self.lmdb_chunk_size, traj_len)

                # chunk = serialize_object(self.worker_traj_history[worker_idx][chunk_start:chunk_end])
                # serialized_chunk = json.dumps(chunk, indent=2).encode()
                serialized_chunk = pickle.dumps(self.worker_traj_history[worker_idx][chunk_start:chunk_end])
                chunks.append(serialized_chunk)

            # Write indices
            txn = self.indices_lmdb_env.begin(write = True)
            traj_info = {"num_chunks": len(chunks), "accomplishments": accomplishments, "horizon": traj_len}
            serialized_traj_info = json.dumps(traj_info, indent=2).encode()
            txn.put(traj_name.encode(), serialized_traj_info)
            txn.commit()

            # Write chunks
            txn = self.trajs_lmdb_env.begin(write = True)
            for i, chunk in enumerate(chunks):
                key = traj_name + "_" + str(i)
                txn.put(key.encode(), chunk)
            txn.commit()

            print("Dumped 1 trajectory of length {} -".format(traj_len), accomplishments)

            self.total_n_recorded_eps += 1

        # clean up
        self.worker_traj_history[worker_idx] = None

    def _broadcast_message(self, command, args = None):
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, command, args = args)

    def _send_message(self, worker_idx, command, args = None):
        self.pipes[worker_idx].send((command, args))

    def _recv_message_nonblocking(self, worker_idx):
        if not self.pipes[worker_idx].poll():
            return None, None

        command, args = self.pipes[worker_idx].recv()

        return command, args