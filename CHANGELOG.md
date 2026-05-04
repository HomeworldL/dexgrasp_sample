# CHANGELOG

## 2026-05-03

### Added
- Added native object asset support in objdata preparation, with `native/` stored as a peer of `scaleXXX` assets.
- Added `run_shape_cluster.py` and `src/shape_cluster.py` for object-level AE feature extraction and KMeans clustering.
- Added `build_dataset_splits_rl.py` for RL-only train/test manifests based on objdata assets and shape-cluster metadata.
- Added `vis_shape_cluster.py` for per-cluster thumbnail visualization.
- Added `print_dataset_objects.py` for lightweight inspection of flattened `DatasetObjects` entries.
- Added root quick-reference docs for shape clustering and RL split.

### Changed
- Refactored `prepare_object_assets.py` to prepare objdata directly from manifests without depending on `DatasetObjects`.
- Refactored `src/dataset_objects.py` back to a read-only indexer; asset generation no longer happens during indexing.
- Extended config support with `build_native_asset`, `use_native_asset`, and `shape_cluster` sections.
- Updated YCB asset config to include default shape-cluster settings (`scale120`, AE-128, `k=4`).
- Updated visualization and sampling helper scripts to respect native asset indexing where applicable.
- Reorganized `AGENTS.md`, `AGENTS_zh.md`, `README.md`, and `README_zh.md` around the current workflow: dataset preparation first, explicit `objdata_*` / `graspdata_*` separation, and a dedicated clustering/RL metadata section.

### Notes
- Current RL shape clustering is based on standard scaled assets only and excludes `native`.
- Current RL split expands object-level cluster metadata onto object-scale records under `datasets/objdata_*/_meta/rl_split/`.
