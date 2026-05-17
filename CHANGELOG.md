# CHANGELOG

## 2026-05-17

### Added
- Added YCB MJWarp GPU configs for scaled and native liberhand-right grasp harvesting with separate `graspdata_*_gpu` output tags.
- Added a prototype `run_bucket_mjw.py` multi-object MJWarp bucket sampler with active object slots and per-object grasp writers.

### Changed
- Moved the single-object MJWarp grasp-close target selection and Jacobian solve path onto GPU tensors while preserving the CPU-aligned anchor/Jacobian semantics.
- Extended the bucket MJWarp sampler with GPU tensor grasp-close and streaming candidate batches so per-object caps stop later candidate allocation.
- Aligned MJWarp grasp outputs with the CPU dataset schema by writing `qpos_squeeze`, `grasp_fail.h5`, and `grasp_fail.npy`, and by applying `fail_keep_ratio` to GPU failure exports.
- Updated MJWarp extforce validation to rebuild pregrasp from the stored squeeze pose plus prepared joints, and to apply hand root stabilization consistently with the CPU path.
- Standardized MJWarp runtime defaults around `batch_size=256`, `nconmax=32`, `naconmax=16384`, `njmax=200`, and `ccd_iterations=200`.
- Aligned the MJWarp bucket sampler's grasp-close stage with the single-object GPU path by using per-slot object point clouds and anchor/Jacobian close semantics.
- Enabled forced USD conversion in the DGN asset config and normalized YCB liberhand-right data config ordering.

## 2026-05-16

### Changed
- Updated objdata asset preparation to carry optional `visual.obj`, `textured_visual.mtl`, and `textured_visual.png` through shared raw and normalized mesh directories.
- Added visual geoms to generated object MJCF assets while keeping visual meshes non-contact and collision meshes in group 3.
- Simplified `prepare_object_assets.py` and `ScaleDatasetBuilder` around source-directory inputs, centralized asset filenames, and shared path/entry helpers.

## 2026-05-12

### Changed
- Refactored objdata preparation to use object-level shared `meshes/` and `meshes_normalized/` directories with thin per-scale asset directories, and updated downstream sampling, rendering, USD conversion, and visualization entrypoints to follow the shared-mesh layout.
- Simplified `DatasetObjects` around the shared-mesh objdata format, removed stale per-scale mesh assumptions, and tightened helper naming and indexing behavior for current asset consumers.
- Standardized config-access helpers in `utils/utils_file.py` around explicit `*_cfg` naming and updated run/asset/sweep entrypoints plus config files to match the new helper and `data.run_scales` conventions.
- Expanded command/documentation coverage with `cmd.md`, refreshed README/AGENTS pipeline examples, and removed stale references to deleted visualization tooling.
- Cleaned unused dataset helper APIs and removed dead code such as the unused `src/fc_metric.py` module.

### Fixed
- Restored `usd_convert` config loading for USD conversion entrypoints.
- Tightened Warp render resume checks so missing camera extrinsics no longer cause false `skip_existing` hits.
- Relaxed objdata indexing so non-USD workflows no longer require `object.urdf` to be present.

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

## 2026-05-05

### Added
- Added `demo_usd.py` for single-asset IsaacLab MJCF-to-USD conversion with direct USD collision schema inspection.
- Added `prepare_object_usds.py` for single-process batch conversion of prepared `object.xml` assets into USD outputs.

### Changed
- Updated prepared object XML internal naming to include the scale/native tag so MuJoCo names remain unique in multi-object scenes.
- Updated USD conversion flow to write `object.usd` directly beside each prepared `object.xml`, relying on IsaacLab-generated `configuration/`, `config.yaml`, and `.asset_hash` files instead of repo-owned metadata sidecars.
- Updated `prepare_object_usds.py` to follow the same config-first asset enumeration as `prepare_object_assets.py`, and to verify that object-level MJCF mass/inertia are preserved on the USD rigid-body root.
- Updated `prepare_object_usds.py` progress reporting to keep tqdm stable, show the current `object_scale_key`, and honor `force=false` by reusing existing USD outputs instead of reconverting every asset.
- Extended RL split manifests with `mjcf_path` and `usd_path`, and reordered shape-cluster ids by ascending distance from each cluster center to the global feature center.
- Simplified shape-cluster artifacts by keeping only labels/history/model metadata, renamed cluster/object label JSON files, removed saved cluster centers, and updated RL split outputs to use `train_object`/`test_object` plus `train_cluster`/`test_cluster` with all scales retained under each cluster.
- Restructured RL split payloads to be object-centric instead of scale-flat, moving shared cluster metadata to the object level and storing per-scale asset paths under each object's `scales` list.
- Refreshed `AGENTS.md`, `AGENTS_zh.md`, `README.md`, and `README_zh.md` to document the current objdata asset layout, in-place USD outputs, and the slimmed shape-cluster / RL-split schemas.

### Docs
- Added an RL-facing usage guide covering prepared asset structure, relative USD paths, shape-cluster labels, and RL split manifests.

## 2026-05-09

### Changed
- Refined RL split generation in `build_dataset_splits_rl.py` with stricter config validation, clearer object/cluster grouping outputs, and improved metadata consistency checks for objdata assets.
- Updated shape-cluster training and visualization flow (`run_shape_cluster.py`, `src/shape_cluster.py`, `vis_shape_cluster.py`) to align labels, simplify artifacts, and reduce ambiguity in downstream RL manifest consumption.
- Improved MuJoCo simulation guards in `src/mj_ho.py` and `src/mjw_ho.py` for more robust contact/stability handling during grasp filtering and replay-related checks.
- Tightened scale dataset utilities (`src/scale_dataset_builder.py`, `utils/utils_file.py`) and expanded corresponding test coverage in `test/test_scale_dataset_builder.py`.
- Updated asset configs (`configs/assets_YCB.json`, `configs/assets_DGN.json`) to match the current clustering/split pipeline expectations.
- Synced workflow documentation (`README.md`, `README_zh.md`, `AGENTS.md`, `AGENTS_zh.md`) with the current objdata/graspdata + USD + RL metadata pipeline.
- Refactored `prepare_object_assets.py` to use object-level source checks, scale-level skip/cleanup behavior, and a minimal objdata manifest containing only successful objects with `name` and `scales_available`.
- Refactored `src/dataset_objects.py` into a pure objdata reader that no longer depends on source-raw manifest fields or `ScaleDatasetBuilder`, and updated all call sites accordingly.
- Switched URDF export for USD conversion to visual/collision mesh parity on `manifold.obj` (`<static>false</static>`, no extra world/floating joint block), and aligned `prepare_object_usds.py` with current IsaacLab `UrdfConverterCfg` fields (`collider_type`, `joint_drive=None`, `collision_from_visuals`).
- Removed `usd_convert.enabled` from asset config handling and exposed URDF conversion controls through `backend`, `merge_joints`, `fix_base`, `make_instanceable`, and `convex_decompose_mesh`.
- Simplified dataset/sample pipeline internals across `run*`, `src/*`, and `utils/*` modules for cleaner control flow and more reproducible behavior under config-driven execution.
- Updated `DatasetObjects` tests to match the minimized objdata manifest schema (`name` + `scales_available`) and made Open3D-dependent coverage explicitly skippable when the dependency is absent.
