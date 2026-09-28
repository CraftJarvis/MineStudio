from gymnasium.spaces.dict import Dict as DictSpace

class HierarchicalActionSpace(DictSpace):
    def __init__(self, *,
                 action_mapper,
                 **kwargs
    ) -> None:
        self.action_mapper = action_mapper
        super().__init__(**kwargs)