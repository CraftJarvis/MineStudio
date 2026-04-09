from minestudio.simulator.minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minestudio.simulator.minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS, ALL_ITEMS
from minestudio.simulator.minerl.herobraine.hero.handlers.translation import TranslationHandler
from minestudio.simulator.minerl.herobraine.hero.handler import Handler
from minestudio.simulator.minerl.herobraine.hero.mc import INVERSE_KEYMAP, SIMPLE_KEYBOARD_ACTION
import minestudio.simulator.minerl.herobraine.hero.handlers as handlers
from minestudio.simulator.minerl.herobraine.hero import handlers as H, mc
from typing import List

import minestudio.simulator.minerl.herobraine
import minestudio.simulator.minerl.herobraine.hero.handlers as handlers
from minestudio.simulator.minerl.herobraine.env_spec import EnvSpec


class HumanSurvival(HumanControlEnvSpec):
    def __init__(self, *args, load_filename = None, inventory = None, preferred_spawn_biome = None, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "MineRLHumanSurvival-v0"
        self.load_filename = load_filename
        self.inventory = inventory
        self.preferred_spawn_biome = preferred_spawn_biome
        super().__init__(
            *args, **kwargs
        )

    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            handlers.EquippedItemObservation(
                items=ALL_ITEMS,
                mainhand=True,
                offhand=True,
                armor=True,
                _default="air",
                _other="air",
            ),
            # handlers.ObservationFromLifeStats(),
            # handlers.ObservationFromCurrentLocation(),
            handlers.ObserveFromFullStats("use_item"),
            # handlers.ObserveFromFullStats("drop"),
            handlers.ObserveFromFullStats("pickup"),
            handlers.ObserveFromFullStats("break_item"),
            handlers.ObserveFromFullStats("craft_item"),
            handlers.ObserveFromFullStats("mine_block"),
            handlers.ObserveFromFullStats("damage_dealt"),
            handlers.ObserveFromFullStats("custom"),
            # handlers.ObserveFromFullStats("entity_killed_by"),
            handlers.ObserveFromFullStats("kill_entity"),
            # handlers.ObserveFromFullStats(None),
        ]

    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        actionables = [
            H.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items()
            if k in SIMPLE_KEYBOARD_ACTION
        ] + [
            H.KeybasedCommandAction(f"hotbar.{i}", f"{i}")
            for i in range(1, 10)
        ] + [
            H.CameraAction()
        ] + [
            H.ChatAction()
        ] + [
            H.VoxelAction(), H.MobAction()
        ]
        return actionables

    def create_rewardables(self) -> List[Handler]:
        return []

    # def create_agent_start(self) -> List[Handler]:
    #     retval = super().create_agent_start()
    #     if self.load_filename is not None:
    #         retval.append(handlers.LoadWorldAgentStart(self.load_filename))
    #     if self.inventory is not None:
    #         retval.append(handlers.InventoryAgentStart(self.inventory))
    #     if self.preferred_spawn_biome is not None:
    #         retval.append(handlers.PreferredSpawnBiome(self.preferred_spawn_biome),)
    #     return retval

    def create_agent_handlers(self) -> List[Handler]:
        return []

    # def create_server_world_generators(self) -> List[Handler]:
    #     if self.load_filename is not None:
    #         return []
    #     return [handlers.DefaultWorldGenerator(force_reset=True)]


    # load world folder?
    def create_agent_start(self) -> List[Handler]:
        retval = super().create_agent_start()
        if self.inventory is not None:
            retval.append(handlers.InventoryAgentStart(self.inventory))
        if self.preferred_spawn_biome is not None:
            retval.append(handlers.PreferredSpawnBiome(self.preferred_spawn_biome))
        return retval

    def create_server_world_generators(self) -> List[Handler]:
        print("loading:", repr(self.load_filename))
        if self.load_filename is not None:
            return [handlers.FileWorldGenerator(
                self.load_filename,
                force_reset=True,
                destroy_after_use=False
            )]
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            # handlers.ServerQuitFromTimeUp((EPISODE_LENGTH * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return True

    def is_from_folder(self, folder: str) -> bool:
        return True

    def get_docstring(self):
        return ""
