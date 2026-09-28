"""Transitional v1 namespace; prefer minestudio.envs for new code."""
__all__ = ["MinecraftSim"]

def __getattr__(name):
    if name == "MinecraftSim":
        from minestudio.simulator.entry import MinecraftSim
        return MinecraftSim
    raise AttributeError(name)
