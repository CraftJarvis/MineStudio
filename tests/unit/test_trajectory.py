import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import Episode, Transition


def observation(value):
    return {"image": np.full((2, 3, 3), value, np.uint8)}


def test_episode_rejects_discontinuous_or_unknown_end():
    initial = observation(0)
    transition = Transition(initial, noop_action(), 0, observation(1), False, False)
    with pytest.raises(ValueError, match="known boundary"):
        Episode(initial, (transition,))
    final = Transition(observation(2), noop_action(), 0, observation(3), True, False)
    with pytest.raises(ValueError, match="temporally aligned"):
        Episode(initial, (transition, final))
    final = Transition(observation(1), noop_action(), 0, observation(2), True, False)
    episode = Episode(initial, (transition, final))
    assert len(episode.observations) == 3
