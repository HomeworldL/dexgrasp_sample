#!/usr/bin/env python3
"""Pretty-print DatasetObjects entries for a run config."""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List

from src.dataset_objects import DatasetObjects
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    data_verbose_from_config,
    generated_dataset_root_from_config,
    graspdata_tag_from_config,
    load_config,
    objdata_tag_from_config,
    raw_dataset_name_from_config,
    raw_dataset_root_from_config,
    run_scales_from_config,
    use_native_asset_from_config,
)


def _format_scale(value: object) -> str:
    if value is None:
        return "native"
    return f"{float(value):.3f}"


def _filter_entries(
    entries: Iterable[Dict],
    object_name: str | None,
    key_substr: str | None,
    native_only: bool,
) -> List[Dict]:
    out: List[Dict] = []
    for entry in entries:
        if object_name and str(entry["object_name"]) != object_name:
            continue
        if key_substr and key_substr not in str(entry["object_scale_key"]):
            continue
        if native_only and not bool(entry.get("is_native", False)):
            continue
        out.append(entry)
    return out


def _print_entries(entries: List[Dict]) -> None:
    if not entries:
        print("No entries matched.")
        return

    headers = ["gid", "object_name", "scale_tag", "scale", "is_native", "object_scale_key"]
    rows = []
    for entry in entries:
        rows.append(
            [
                str(entry["global_id"]),
                str(entry["object_name"]),
                str(entry.get("scale_tag", "")),
                _format_scale(entry.get("scale")),
                str(bool(entry.get("is_native", False))),
                str(entry["object_scale_key"]),
            ]
        )

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print DatasetObjects entries.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("--object-name", type=str, default=None, help="Filter exact object name.")
    parser.add_argument("--key-substr", type=str, default=None, help="Filter object_scale_key substring.")
    parser.add_argument("--native-only", action="store_true", help="Show only native entries.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = DatasetObjects(
        raw_dataset_root=raw_dataset_root_from_config(cfg),
        raw_dataset_name=raw_dataset_name_from_config(cfg),
        scales=run_scales_from_config(cfg),
        objdata_tag=objdata_tag_from_config(cfg, args.config),
        include_native=use_native_asset_from_config(cfg),
        graspdata_tag=graspdata_tag_from_config(cfg, args.config),
        generated_dataset_root=generated_dataset_root_from_config(cfg),
        verbose=data_verbose_from_config(cfg),
    )

    entries = sorted(ds.get_entries(), key=lambda item: int(item["global_id"]))
    entries = _filter_entries(
        entries,
        object_name=args.object_name,
        key_substr=args.key_substr,
        native_only=bool(args.native_only),
    )
    print(
        f"[print_dataset_objects] entries={len(entries)} "
        f"config={args.config} include_native={use_native_asset_from_config(cfg)}"
    )
    _print_entries(entries)


if __name__ == "__main__":
    main()
