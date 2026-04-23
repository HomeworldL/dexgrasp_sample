import pytest

from utils.utils_file import hand_root_stabilization_from_config


def test_hand_root_stabilization_from_config_parses_valid_block():
    cfg = {
        "hand": {
            "root_stabilization": {
                "root_body_name": "hand_right_palm",
                "root_scale": 1e9,
            }
        }
    }
    out = hand_root_stabilization_from_config(cfg)
    assert out == {
        "root_body_name": "hand_right_palm",
        "root_scale": 1e9,
    }


def test_hand_root_stabilization_from_config_returns_none_when_absent():
    cfg = {"hand": {}}
    assert hand_root_stabilization_from_config(cfg) is None


def test_hand_root_stabilization_from_config_rejects_invalid_values():
    cfg_bad_scale = {
        "hand": {
            "root_stabilization": {
                "root_body_name": "hand_right_palm",
                "root_scale": 0.0,
            }
        }
    }
    with pytest.raises(ValueError):
        hand_root_stabilization_from_config(cfg_bad_scale)

    cfg_bad_name = {
        "hand": {
            "root_stabilization": {
                "root_body_name": "",
                "root_scale": 1e9,
            }
        }
    }
    with pytest.raises(ValueError):
        hand_root_stabilization_from_config(cfg_bad_name)
