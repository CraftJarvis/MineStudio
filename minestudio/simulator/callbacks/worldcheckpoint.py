from pathlib import Path
import shutil
import json
import time
import re
from typing import Optional

from minestudio.simulator.callbacks.callback import MinecraftCallback


class WorldCheckpointCallback(MinecraftCallback):
    """
    在 Python 外层导出 Minecraft world checkpoint。

    新逻辑：
    - 通过 socket 给 Java 发 <SaveWorld/>
    - 再发 <GetWorldPath/>
    - 直接使用 Java 返回的真实 world_path
    - 不再依赖扫描 working_dir/saves 猜测当前 world
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

    def before_reset(self, sim, reset_flag):
        return reset_flag

    def after_reset(self, sim, obs, info):
        self.current_step = 0
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

        # 1) 先请求 Java 保存
        comms.send_message(inst.client_socket, b"<SaveWorld/>")
        reply = comms.recv_message(inst.client_socket)
        self._log(f"[Checkpoint] SaveWorld reply: {reply!r}")

        # 2) 推几帧，让 Java 里的 pendingSave 真正执行
        # 你原来这里写了 30 次 step，这是合理的保守做法
        for _ in range(10):
            sim.step(sim.noop_action(), no_callback=True)
            time.sleep(0.05)

        # 3) 向 Java 请求真实 world path
        world_info = self._request_world_info(sim)
        self.last_world_info = world_info

        world_path = Path(world_info["world_path"])
        world_name = world_info["world_name"]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        #export_name = f"{world_name}_{tag or 'manual'}_{timestamp}"
        export_dir = self.export_root #/ export_name
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

    def _request_world_info(self, sim):
        from minestudio.simulator.minerl.env import comms

        inst = self._get_instance(sim)

        comms.send_message(inst.client_socket, b"<GetWorldPath/>")
        reply = comms.recv_message(inst.client_socket)

        if isinstance(reply, bytes):
            text = reply.decode("utf-8", errors="replace")
        else:
            text = str(reply)

        self._log(f"[Checkpoint] GetWorldPath reply: {text}")

        world_root = self._extract_xml_tag(text, "WorldRoot")
        region_root = self._extract_xml_tag(text, "RegionRoot")

        if not world_root:
            raise RuntimeError(f"WorldRoot missing in GetWorldPath reply: {text}")
        if not region_root:
            raise RuntimeError(f"RegionRoot missing in GetWorldPath reply: {text}")

        world_path = Path(world_root).parent

        if not world_path.exists():
            raise FileNotFoundError(f"world path returned by Java does not exist: {world_path}")

        inst = self._get_instance(sim)
        working_dir = Path(inst.working_dir)

        import pdb
        pdb.set_trace()

        return {
            "working_dir": str(working_dir),
            "world_name": world_path.name,
            "world_path": str(world_path),
            "instance_uuid": getattr(inst, "uuid", None),
            "instance_id": getattr(inst, "instance_id", None),
            "port": getattr(inst, "_port", None),
        }

    def _extract_xml_tag(self, text: str, tag: str) -> Optional[str]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, text, flags=re.DOTALL)
        if not m:
            return None
        return m.group(1).strip()

    def _get_instance(self, sim):
        base_env = getattr(sim, "env", None)
        if base_env is None:
            raise RuntimeError("sim.env is missing")

        instances = getattr(base_env, "instances", None)
        if instances is None:
            raise RuntimeError("sim.env.instances is missing")

        if not isinstance(instances, list) or len(instances) == 0:
            raise RuntimeError(f"Unexpected instances: {instances}")

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