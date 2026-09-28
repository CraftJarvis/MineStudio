import pytest

from minestudio.policies.vpt import VPTConfig


def test_legacy_options_preserve_context_length_and_normalization():
    config = VPTConfig.from_legacy_kwargs(
        {
            "hidsize": 64,
            "img_shape": [128, 128, 3],
            "impala_width": 2,
            "attention_memory_size": 256,
            "timesteps": 128,
            "n_recurrence_layers": 4,
            "init_norm_kwargs": {"batch_norm": False, "group_norm_groups": 1},
            "impala_kwargs": {"post_pool_groups": 1},
            "use_pre_lstm_ln": False,
            "only_img_input": True,
        }
    )
    kwargs = config.network_kwargs()
    assert kwargs["attention_memory_size"] - kwargs["timesteps"] == 128
    assert kwargs["init_norm_kwargs"]["group_norm_groups"] == 1
    assert kwargs["use_pre_lstm_ln"] is False
    assert kwargs["impala_kwargs"]["post_pool_groups"] == 1


def test_legacy_unknown_or_ambiguous_options_are_rejected():
    with pytest.raises(ValueError, match="timesteps"):
        VPTConfig.from_legacy_kwargs({"img_shape": [128, 128, 3]})
    with pytest.raises(ValueError, match="Unsupported"):
        VPTConfig.from_legacy_kwargs({"timesteps": 1, "img_shape": [128, 128, 3], "unknown": 1})
    with pytest.raises(ValueError, match="exceed"):
        VPTConfig(sequence_length=128, attention_memory_size=128)
