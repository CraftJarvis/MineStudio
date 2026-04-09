from pathlib import Path
import shutil
import json
import time
from typing import Optional

from minestudio.simulator.callbacks.callback import MinecraftCallback

from pathlib import Path


class WorldCheckpointCallback(MinecraftCallback):
    """
    在 Python 外层导出 Minecraft world checkpoint。

    核心逻辑：
    - 从 sim.env.instances[0].working_dir 获取当前实例 working_dir
    - 在 <working_dir>/saves 下找到真实 world 目录
    - 在 before_close 中抢在 env.close() / working_dir cleanup 前导出
    """

    def __init__(
        self,
        export_root: str,
        auto_export_on_close: bool = True,
        export_every_n_steps: Optional[int] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.export_root = Path(export_root)
        self.auto_export_on_close = auto_export_on_close
        self.export_every_n_steps = export_every_n_steps
        self.overwrite = overwrite
        self.verbose = verbose

        self.current_step = 0
        self.last_export_path = None
        self.last_world_info = None

        self._worlds_before_reset = set()
        self.active_world_path = None
        self.active_world_name = None

    def before_reset(self, sim, reset_flag):
        try:
            world_dirs = self._list_world_dirs(sim)
            self._worlds_before_reset = {p.name for p in world_dirs}
            self._log(f"[Checkpoint] worlds before reset: {sorted(self._worlds_before_reset)}")
        except Exception as e:
            self._log(f"[Checkpoint] before_reset scan failed: {e}")
            self._worlds_before_reset = set()
        return reset_flag


    def after_reset(self, sim, obs, info):
        self.current_step = 0
        try:
            world_dirs = self._list_world_dirs(sim)
            after_names = {p.name for p in world_dirs}
            new_names = after_names - self._worlds_before_reset

            chosen = None

            # 情况1：reset 后新增了一个 world
            if len(new_names) == 1:
                new_name = next(iter(new_names))
                chosen = next(p for p in world_dirs if p.name == new_name)
                self._log(f"[Checkpoint] detected newly created world: {new_name}")

            # 情况2：没有明确新增，则选最近修改的
            elif len(world_dirs) > 0:
                chosen = max(world_dirs, key=self._world_mtime)
                self._log(
                    f"[Checkpoint] fallback to most recently modified world: {chosen.name}"
                )

            if chosen is None:
                raise RuntimeError("cannot determine active world after reset")

            self.active_world_path = str(chosen)
            self.active_world_name = chosen.name

            self.last_world_info = self._resolve_world_info_from_path(sim, chosen)
            self._log(f"[Checkpoint] active world set to: {self.last_world_info}")

        except Exception as e:
            self._log(f"[Checkpoint] after_reset resolve failed: {e}")

        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.current_step += 1

        if (
            self.export_every_n_steps is not None
            and self.export_every_n_steps > 0
            and self.current_step % self.export_every_n_steps == 0
        ):
            try:
                export_path = self.export_now(
                    sim,
                    tag=f"step{self.current_step}",
                    live=True,
                )
                self._log(f"[Checkpoint] periodic export -> {export_path}")
            except Exception as e:
                self._log(f"[Checkpoint] periodic export failed: {e}")

        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        if not self.auto_export_on_close:
            return


        export_path = self.export_now(sim, tag="final", live=False)
        self._log(f"[Checkpoint] final export before close -> {export_path}")
        

    # -------------------------
    # public API
    # -------------------------

    def export_now(self, sim, tag=None, live=True) -> Path:

        from minestudio.simulator.minerl.env import comms

        inst = self._get_instance(sim)
        comms.send_message(inst.client_socket, "<SaveWorld/>".encode("utf-8"))
        reply = comms.recv_message(inst.client_socket)
        for i in range(30):
            sim.step(sim.noop_action(), no_callback=True)  # 确保 SaveWorld 消息被处理
            time.sleep(0.1) #应该是没用的

        print(f"[Checkpoint] SaveWorld reply: {reply.decode('utf-8')}")

        if self.active_world_path is not None:
            world_path = Path(self.active_world_path)
            world_info = self._resolve_world_info_from_path(sim, world_path)
        else:
            # 没缓存到时再 fallback
            world_info = self._resolve_world_info_fallback(sim)

        self.last_world_info = world_info

        world_path = Path(world_info["world_path"])
        world_name = world_info["world_name"]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_name = f"{world_name}_{tag or 'manual'}_{timestamp}"
        export_dir = self.export_root / export_name
        export_world_dir = export_dir / "world"

        if export_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"Export dir already exists: {export_dir}")
            shutil.rmtree(export_dir)

        export_dir.mkdir(parents=True, exist_ok=True)
        self._copy_world_dir(world_path, export_world_dir)

        meta = {
            "timestamp": timestamp,
            "tag": tag,
            "live_export": bool(live),
            "step": self.current_step,
            "working_dir": world_info["working_dir"],
            "saves_dir": world_info["saves_dir"],
            "world_name": world_info["world_name"],
            "world_path": world_info["world_path"],
            "instance_uuid": world_info["instance_uuid"],
            "instance_id": world_info["instance_id"],
            "port": world_info["port"],
        }

        with open(export_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.last_export_path = export_dir
        return export_dir

    # -------------------------
    # internal helpers
    # -------------------------
    def _resolve_world_info(self, sim):
        inst = self._get_instance(sim)

        working_dir = Path(inst.working_dir)
        saves_dir = working_dir / "saves"

        if not saves_dir.exists():
            raise FileNotFoundError(f"saves dir not found: {saves_dir}")

        world_dirs = []
        for p in saves_dir.iterdir():
            if p.is_dir() and (p / "level.dat").exists():
                world_dirs.append(p)

        if len(world_dirs) == 0:
            raise FileNotFoundError(f"No valid world found under {saves_dir}")

        if len(world_dirs) > 1:
            raise RuntimeError(
                f"Multiple worlds found under {saves_dir}: {[p.name for p in world_dirs]}"
            )

        world_path = world_dirs[0]

        return {
            "working_dir": str(working_dir),
            "saves_dir": str(saves_dir),
            "world_name": world_path.name,
            "world_path": str(world_path),
            "instance_uuid": getattr(inst, "uuid", None),
            "instance_id": getattr(inst, "instance_id", None),
            "port": getattr(inst, "_port", None),
        }

    def _get_instance(self, sim):
        base_env = getattr(sim, "env", None)
        if base_env is None:
            raise RuntimeError("sim.env is missing")

        instances = getattr(base_env, "instances", None)
        if instances is None:
            raise RuntimeError("sim.env.instances is missing")

        if not isinstance(instances, list) or len(instances) == 0:
            raise RuntimeError(f"Unexpected instances: {instances}")

        # 单 agent 环境，默认用第一个实例
        inst = instances[0]

        if not hasattr(inst, "working_dir"):
            raise RuntimeError("instance has no working_dir")

        return inst

    def _copy_world_dir(self, src: Path, dst: Path):
        if not src.exists():
            raise FileNotFoundError(f"World source not found: {src}")
        if not (src / "level.dat").exists():
            raise RuntimeError(f"Invalid world dir (missing level.dat): {src}")

        if dst.exists():
            shutil.rmtree(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, symlinks=False)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _resolve_world_info_from_path(self, sim, world_path: Path):
        inst = self._get_instance(sim)
        working_dir = Path(inst.working_dir)
        saves_dir = working_dir / "saves"

        return {
            "working_dir": str(working_dir),
            "saves_dir": str(saves_dir),
            "world_name": world_path.name,
            "world_path": str(world_path),
            "instance_uuid": getattr(inst, "uuid", None),
            "instance_id": getattr(inst, "instance_id", None),
            "port": getattr(inst, "_port", None),
        }

    def _resolve_world_info_fallback(self, sim):
        world_dirs = self._list_world_dirs(sim)
        if len(world_dirs) == 0:
            raise FileNotFoundError("No valid world found")

        if self.active_world_name is not None:
            for p in world_dirs:
                if p.name == self.active_world_name:
                    return self._resolve_world_info_from_path(sim, p)

        # fallback: 最近修改的
        chosen = max(world_dirs, key=self._world_mtime)
        return self._resolve_world_info_from_path(sim, chosen)



    def _list_world_dirs(self, sim):
        inst = self._get_instance(sim)
        working_dir = Path(inst.working_dir)
        saves_dir = working_dir / "saves"

        if not saves_dir.exists():
            return []

        world_dirs = []
        for p in saves_dir.iterdir():
            if p.is_dir() and (p / "level.dat").exists():
                world_dirs.append(p)
        return world_dirs

    def _world_mtime(self, world_path: Path) -> float:
        candidates = [world_path]
        level_dat = world_path / "level.dat"
        if level_dat.exists():
            candidates.append(level_dat)
        return max(p.stat().st_mtime for p in candidates if p.exists())

