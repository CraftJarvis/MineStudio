'''
Date: 2024-12-12 16:35:39
LastEditors: zhengxinyue
LastEditTime: 2024-12-12 17:47:25
FilePath: /MineStudio/minestudio/benchmark/utility/record_call.py
'''

import av
from pathlib import Path
from minestudio.simulator.callbacks.callback import MinecraftCallback
from typing import Literal
from rich import print

class RecordCallback(MinecraftCallback):
    def __init__(self, record_path: str, fps: int = 20, frame_type: Literal['pov', 'obs'] = 'pov', recording: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.record_path = Path(record_path)
        self.record_path.mkdir(parents=True, exist_ok=True)
        self.recording = recording
        if recording:
            print(f'[green]Recording enabled, saving episodes to {self.record_path}[/green]')
        self.fps = fps
        self.frame_type = frame_type
        self.episode_id = 0
        self.frames = []
    
    def _get_message(self, info):
        message = info.get('message', {})
        message['RecordCallback'] = f'Recording: {"On" if self.recording else "Off"}, Recording Time: {len(self.frames)}'
        return message

    def before_reset(self, sim, reset_flag: bool) -> bool:
        if self.recording:
            self._save_episode()
            self.episode_id += 1
        return reset_flag

    def after_reset(self, sim, obs, info):
        sim.callback_messages.add("Press 'R' to start/stop recording.")
        # this message would be displayed in the GUI when command mode is on
        info['message'] = self._get_message(info)
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if self.recording:
            if self.frame_type == 'obs':
                self.frames.append(obs['image'])
            elif self.frame_type == 'pov':
                self.frames.append(info['pov'])
            else:
                raise ValueError(f'Invalid frame_type: {self.frame_type}')
        info['message'] = self._get_message(info)
        return obs, reward, terminated, truncated, info
    
    def before_close(self, sim):
        if self.recording:
            self._save_episode()
    
    def _save_episode(self):
        if len(self.frames) == 0:
            return 
        output_path = self.record_path / f'episode_{self.episode_id}.mp4'
        with av.open(output_path, mode="w", format='mp4') as container:
            stream = container.add_stream("h264", rate=self.fps)
            stream.width = self.frames[0].shape[1]
            stream.height = self.frames[0].shape[0]
            for frame in self.frames:
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        print(f'[green]Episode {self.episode_id} saved at {output_path}[/green]')
        self.frames = []