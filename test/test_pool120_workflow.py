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


def test_split_cluster_ids() -> None:
    in_domain_ids, ood_ids = subset_pool.split_cluster_ids(
        [0, 1, 2, 3, 4], in_domain_cluster_max_id=2
    )
    assert in_domain_ids == [0, 1, 2]
    assert ood_ids == [3, 4]


def test_train_test_ood_selection_rules() -> None:
    ordered_cluster_members = {
        "0": ["a0", "a1", "a2", "a3"],
        "1": ["b0", "b1", "b2", "b3"],
        "2": ["c0", "c1", "c2", "c3"],
    }
    in_domain_members = subset_pool.build_cluster_member_lists(
        ordered_cluster_members, [0, 1]
    )
    ood_members = subset_pool.build_cluster_member_lists(ordered_cluster_members, [2])

    train_candidates, test_candidates = subset_pool.split_even_odd_cluster_members(
        in_domain_members
    )
    assert train_candidates == {"0": ["a0", "a2"], "1": ["b0", "b2"]}
    assert test_candidates == {"0": ["a1", "a3"], "1": ["b1", "b3"]}

    train_selected = subset_pool.select_balanced_cluster_members(
        train_candidates, count=2, split_name="train", seed=11
    )
    ordered_train = subset_pool.interleave_cluster_members(train_selected)
    assert ordered_train == ["a0", "b0"]

    records = OrderedDict(
        (name, {"object_name": name})
        for members in ordered_cluster_members.values()
        for name in members
    )
    train_records = subset_pool.select_prefix_records(
        ordered_train, records, count=2, split_name="train"
    )
    assert list(train_records.keys()) == ["a0", "b0"]

    test_shuffled = subset_pool.shuffle_cluster_members(test_candidates, seed=101)
    test_selected = subset_pool.select_balanced_cluster_members(
        test_shuffled, count=2, split_name="test", seed=111
    )
    ordered_test = subset_pool.interleave_cluster_members(test_selected)
    test_records = subset_pool.select_prefix_records(
        ordered_test, records, count=2, split_name="test"
    )
    subtest_records = subset_pool.select_prefix_records(
        ordered_test, records, count=1, split_name="subtest"
    )
    assert len(set(train_records.keys()) & set(test_records.keys())) == 0
    assert list(subtest_records.keys()) == list(test_records.keys())[:1]

    ood_shuffled = subset_pool.shuffle_cluster_members(ood_members, seed=202)
    ood_selected = subset_pool.select_balanced_cluster_members(
        ood_shuffled, count=2, split_name="ood", seed=222
    )
    ordered_ood = subset_pool.interleave_cluster_members(ood_selected)
    ood_records = subset_pool.select_prefix_records(
        ordered_ood, records, count=2, split_name="ood"
    )
    subood_records = subset_pool.select_prefix_records(
        ordered_ood, records, count=1, split_name="subood"
    )
    assert set(ood_records.keys()).issubset({"c0", "c1", "c2", "c3"})
    assert list(subood_records.keys()) == list(ood_records.keys())[:1]


def test_balanced_selection_refills_from_clusters_with_capacity() -> None:
    selected = subset_pool.select_balanced_cluster_members(
        {"0": ["a0"], "1": ["b0", "b1", "b2"], "2": ["c0", "c1", "c2"]},
        count=5,
        split_name="train",
        seed=0,
    )
    assert len(selected["0"]) == 1
    assert sum(len(members) for members in selected.values()) == 5
