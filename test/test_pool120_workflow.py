from __future__ import annotations

from collections import OrderedDict

import build_dataset_subset_pool as subset_pool
import prepare_object_assets_pool as prepare_pool


def test_resolve_pool_filter_cfg_defaults() -> None:
    cfg = {"pool_filter": {"enabled": True, "version": "v1", "mesh_kind": "coacd"}}
    resolved = prepare_pool.resolve_pool_filter_cfg(cfg)
    assert resolved["enabled"] is True
    assert resolved["version"] == "v1"
    assert resolved["mesh_kind"] == "coacd"
    assert resolved["thin_ratio_min"] == 0.12
    assert resolved["flat_ratio_min"] == 0.12


def test_pool_filter_keep_requires_both_ratios() -> None:
    pool_cfg = {
        "enabled": True,
        "version": "v1",
        "mesh_kind": "coacd",
        "thin_ratio_min": 0.12,
        "flat_ratio_min": 0.12,
    }
    assert (
        prepare_pool._pool_filter_keep(  # noqa: SLF001
            {"thin_ratio": 0.13, "flat_ratio": 0.15}, pool_cfg
        )
        is True
    )
    assert (
        prepare_pool._pool_filter_keep(  # noqa: SLF001
            {"thin_ratio": 0.11, "flat_ratio": 0.15}, pool_cfg
        )
        is False
    )
    assert (
        prepare_pool._pool_filter_keep(  # noqa: SLF001
            {"thin_ratio": 0.13, "flat_ratio": 0.11}, pool_cfg
        )
        is False
    )


def test_subset_selection_round_robin_and_prefix() -> None:
    ordered_cluster_members = {
        "0": ["obj_a", "obj_b", "obj_c"],
        "1": ["obj_d", "obj_e"],
    }
    train_members, test_members = subset_pool.split_cluster_members_for_pool(
        ordered_cluster_members
    )
    assert train_members == {"0": ["obj_a", "obj_c"], "1": ["obj_d"]}
    assert test_members == {"0": ["obj_b"], "1": ["obj_e"]}

    ordered_train = subset_pool.interleave_cluster_members(train_members)
    ordered_test = subset_pool.interleave_cluster_members(test_members)
    assert ordered_train == ["obj_a", "obj_d", "obj_c"]
    assert ordered_test == ["obj_b", "obj_e"]

    records = OrderedDict(
        (
            ("obj_a", {"object_name": "obj_a"}),
            ("obj_b", {"object_name": "obj_b"}),
            ("obj_c", {"object_name": "obj_c"}),
            ("obj_d", {"object_name": "obj_d"}),
            ("obj_e", {"object_name": "obj_e"}),
        )
    )
    train_subset = subset_pool.select_prefix_records(
        ordered_train, records, count=2, split_name="train"
    )
    test_subset = subset_pool.select_prefix_records(
        ordered_test, records, count=1, split_name="test"
    )
    assert list(train_subset.keys()) == ["obj_a", "obj_d"]
    assert list(test_subset.keys()) == ["obj_b"]
