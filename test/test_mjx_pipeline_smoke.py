import json
from pathlib import Path


def test_mjx_config_block_exists():
    cfg_path = Path("configs/run_YCB_liberhand.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "mjx" in cfg
    assert cfg["mjx"]["impl"] == "jax"
    assert "batch" in cfg["mjx"]


def test_mjx_import_or_skip():
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
    except Exception as exc:
        import pytest

        pytest.skip(f"MJX stack is unavailable in current env: {exc!r}")
