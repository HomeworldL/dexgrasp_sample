# dexgrasp_sample

MuJoCo-based dexterous grasp sampling pipeline for multi-dataset 3D object assets.

## 1. Project Overview
This repository generates offline grasp configurations for object meshes.

Mainline workflow:
1. Build object-scale assets from processed meshes.
2. Sample object surface points and normals.
3. Sample grasp frame candidates and convert to hand base pose.
4. Run MuJoCo staged collision filtering + grasp closing + external-force validation.
5. Save valid grasps to `grasp.h5`.
6. Export `grasp.npy` from `grasp.h5`.
7. (Optional, post-process) Render partial point clouds per object-scale via Warp.

Mainline code (from `AGENTS.md`):
- `run.py`
- `run_multi.py`
- `run_warp_render.py`
- `src/mj_ho.py`
- `src/sample.py`
- `src/dataset_objects.py`

---

## 2. Dependencies
Install base dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` currently includes:
- `numpy`, `scipy`, `torch`
- `trimesh`, `open3d`
- `mujoco`, `mujoco-viewer`
- `h5py`, `tqdm`, `transforms3d`
- `viser`, `pytest`

Optional dependencies by feature:
- Warp partial point cloud rendering:
  - `warp-lang`
  - headless EGL/OpenGL-capable environment
- Plotly grasp frame visualization:
  - `plotly`

---

## 3. Repository Structure and Major Files
### 3.1 Pipeline Entrypoints
- `run.py`
  - Single object-scale grasp sampling worker.
  - Requires explicit object-scale inputs (`--object-scale-key`, `--coacd-path`, `--mjcf-path`, `--output-dir`).
- `run_multi.py`
  - Parallel dispatcher for all object-scales from config datasets.
  - Spawns subprocesses that call `run.py`.
- `run_warp_render.py`
  - Post-grasp partial point cloud rendering for object-scales using Warp.
  - Parallelized by object-scale entries (not by camera-view splitting across workers).

### 3.2 Core Modules
- `src/dataset_objects.py`
  - Manifest-driven merged object-scale index.
  - Reads `manifest.process_meshes.json`, keeps only `process_status=success`.
  - Prebuilds scaled assets under `datasets/<dataset_tag>/<object>/scaleXXX/`.
  - `dataset_tag` is derived from config stem by replacing leading `run_` with `graspdata_`.
- `src/scale_dataset_builder.py`
  - Builds per-scale assets:
    - `coacd.obj`
    - `convex_parts/*.obj`
    - `object.xml`
- `src/sample.py`
  - Grasp frame sampling and point cloud FPS utilities.
- `src/mj_ho.py`
  - MuJoCo hand-object simulation helper:
    - contact checks
    - grasp closing simulation
    - external-force validation

### 3.3 Visualization Entrypoints (root)
- `vis_obj.py`
  - Visualize object mesh + sampled point cloud in Viser, then open MuJoCo paused viewer.
- `vis_ho.py`
  - Visualize hand-object posed meshes in Viser, then open MuJoCo paused viewer.
- `vis_grasp.py`
  - Visualize object + selected grasp hand meshes (4 qpos stages) in Viser.
  - Also visualize all grasp frames in Plotly.
- `vis_partial_pc.py`
  - Visualize saved Warp partial point clouds (`partial_pc_XX.npy` world, `partial_pc_cam_XX.npy` camera) for one object-scale.
  - Optional camera frame display from `cam_ex_XX.npy`.

---

## 4. Dataset Interface and Output Layout
### 4.1 Input dataset root
Configured by JSON field `dataset.root` (recommended):
- `assets/objects/processed`

The repository commonly uses a symlink:
- `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`

### 4.2 Manifest gating
Only manifest entries with:
- `process_status=success`
are eligible.

### 4.3 Object-scale flattened indexing
`DatasetObjects` provides merged global indexing over dataset list and scales.

Each entry has fields:
- `global_id`
- `object_name`
- `coacd_abs`
- `convex_parts_abs`
- `scale`
- `mjcf_abs`
- `output_dir_abs`
- `object_scale_key`

### 4.4 Generated filesystem layout
For each object-scale:

```text
datasets/<dataset_tag>/<object_name>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/part_000.obj ...
  grasp.h5
  grasp.npy
  <warp_render.output_subdir>/
    cam_in.npy
    cam_ex_00.npy ...
    partial_pc_00.npy ...
    partial_pc_cam_00.npy ...
```

Notes:
- `run.py` always writes `grasp.h5` and exports `grasp.npy`.
- Point cloud outputs are kept separate from grasp arrays.

---

## 5. Grasp Dataset Format (`grasp.h5`)
Metadata datasets in H5:
- `object_name` (string)
- `scale` (float32)
- `hand_name` (string)
- `rot_repr` (`wxyz+qpos`)

Main qpos datasets inside H5:
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`

Each row is:
- `[tx, ty, tz, qw, qx, qy, qz, q1...qN]`

Semantics:
- `qpos_prepared`: base pose + prepared joints
- `qpos_approach`: base pose + approach joints
- `qpos_init`: approach joints + shifted pose for approach start
- `qpos_grasp`: final grasp pose after closing

---

## 6. CLI Usage (Main Entrypoints)
### 6.1 `run.py`
Single object-scale grasp generation.

```bash
python run.py \
  -c configs/run_YCB_liberhand.json \
  --object-scale-key YCB_002_master_chef_can__scale080 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale080/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale080/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale080 \
  -v
```

Args:
- `--object-scale-key` required
- `--coacd-path` required
- `--mjcf-path` required
- `--output-dir` required
- `object_name` and `scale` are parsed from `--object-scale-key`
- `-c/--config` optional (default `configs/run_YCB_liberhand.json`)
- `-v/--verbose` optional

### 6.2 `run_multi.py`
Parallel all object-scale entries in config.

```bash
python run_multi.py -c configs/run_YCB_liberhand.json -j 4 -v
```

Args:
- `-j/--max-parallel` max subprocesses
- `--script` script path called per entry (default `run.py`)
- `-c/--config` config path
- `-v/--verbose`

After parallel execution finishes, `run_multi.py` also writes:
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

Build rules:
- Re-scan all object-scale entries for the current config.
- Only include complete outputs: `grasp.h5`, `grasp.npy`, `cam_in.npy`, and matched `partial_pc_XX.npy`, `partial_pc_cam_XX.npy`, `cam_ex_XX.npy`.
- Split by `object_name` with a stable 4:1 train/test partition, so all scales of the same object stay in the same split.
- Even when every object-scale already exists and no parallel sampling is launched, this split export step still runs.

Each JSON item remains flattened at object-scale granularity, with all paths relative to `datasets/<dataset_tag>/`. Fields:
- `global_id`
- `object_scale_key`
- `object_name`
- `output_path`
- `coacd_path`
- `mjcf_path`
- `grasp_h5_path`
- `grasp_npy_path`
- `partial_pc_path`
- `partial_pc_cam_path`
- `cam_ex_path`
- `cam_in`
- `scale`

### 6.3 `run_warp_render.py`
Post-process partial point cloud rendering.

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -j 1 -v
```

Single entry examples:

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -i 0
python run_warp_render.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

Args:
- `-c/--config`
- `-i/--obj-id` or `-k/--obj-key`
- `-j/--max-parallel`
- `--gpu-lst` override device list, e.g. `0,1` or `cpu`
- `--force` ignore skip-existing
- `-v/--verbose`

Parallel semantics:
- Worker granularity is object-scale entries.
- One entry renders `n_cols * n_rows` views in one worker.

### 6.4 `eval_dataset.py`
Evaluate the existing `qpos_grasp` entries already stored in `train.json` or `test.json`, without using a network.

```bash
python eval_dataset.py \
  -c configs/run_YCB_liberhand.json \
  --split test \
  --visualize \
  -v
```

Notes:
- The split manifest is resolved automatically from `config` and `--split` as `datasets/<dataset_tag>/<split>.json`.
- Each object-scale loads `qpos_grasp` directly from `grasp.h5`.
- Each `qpos_grasp` is first rejected if `qpos_init` or `qpos_prepared` is already in contact.
- Only grasps that pass the pre-contact checks are evaluated with `MjHO.sim_under_extforce`.
- `--visualize` opens the MuJoCo viewer for both the pre-contact checks and the extforce validation.
- The script reports per-object-scale and whole-split success counts and rates.
- No new grasp artifact files are written.

---

## 7. Visualization Usage
### 7.1 `vis_obj.py`
```bash
python vis_obj.py -c configs/run_YCB_liberhand.json -i 0
# or
python vis_obj.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

### 7.2 `vis_ho.py`
```bash
python vis_ho.py -c configs/run_YCB_liberhand.json -i 0
```

### 7.3 `vis_grasp.py`
```bash
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0 --vis-ids 0,10,-1 --frame-stage qpos_grasp
```

Useful options:
- `--grasp-path` explicit `.h5` / `.npy`
- `--skip-plotly`
- `--plotly-html out.html`

### 7.4 `vis_partial_pc.py`
```bash
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0 --view-ids 0,1 --show-cam-frames
```

Useful options:
- `--pc-subdir` override pointcloud subdir
- `--hide-mesh`
- `--show-cam-frames`

### 7.5 `tools/visualization/plot_grasp_pose_plotly.py`
Run with project root in module path:

```bash
PYTHONPATH=. python tools/visualization/plot_grasp_pose_plotly.py -c configs/run_YCB_liberhand.json -i 0
```

---

## 8. Config JSON Guide
Main config files:
- `configs/run_<dataset_group>_<hand>.json`

Examples:
- `run_YCB_liberhand.json`
- `run_DGN2_liberhand2.json`
- `run_HOPE_inspire.json`

### 8.1 `dataset`
- `root`: dataset root path
- `include`: dataset names list
- `scales`: fixed scale list
- `verbose`: dataset indexing logs

### 8.2 `hand`
- `xml_path`
- `prepared_joints`
- `approach_joints`
- `shift_local`
- `transform`
  - `base_rot_grasp_to_palm`
  - `extra_euler` (`axis`, `degrees`)
- `target_body_params` (contact/distance weights)

### 8.3 `sampling`
- `n_points`: surface sampling points
- `downsample_for_sim`: object point count used in simulation helper
- `Nd`, `rot_n`, `d_min`, `d_max`, `max_points`: grasp frame sampling controls

### 8.4 `sim_grasp`
- MuJoCo grasp-closing step settings (`Mp`, `steps`, `speed_gain`, `max_tip_speed`)
- `contact_min_count`: minimum hand-object contact count before external-force validation

### 8.5 `extforce`
- External-force validation settings (`duration`, thresholds, force magnitude, check interval)
- `run_mjw.py` stops the extforce stage early when either:
  - `output.max_cap` valid grasps have been written
  - extforce wall-clock time exceeds `output.max_time_sec`
- `run.py` treats `output.max_time_sec` as a total wall-clock budget for the whole CPU sampling run
- Current default recommendation for Warp configs is `output.max_time_sec = 90.0`

### 8.6 `output`
- `max_cap`, `max_time_sec`, `h5_name`, `npy_name`
- `dataset_root` is optional and used by dataset-level helper scripts such as `run_multi.py`
- Current implementation writes `grasp.h5` in each `output_dir_abs`.
- Current implementation always exports `grasp.npy` from `grasp.h5`.

### 8.7 `warp_render`
- Device and parallel:
  - `gpu_lst`, `thread_per_gpu`
- Output and skip:
  - `output_subdir` (default now `partial_pc_warp`)
  - `save_pc/save_rgb/save_depth`
  - `skip_existing`
- Rendering geometry:
  - `tile_width`, `tile_height`, `n_cols`, `n_rows`, `z_near`, `z_far`
- Camera model:
  - `intrinsics` (`preset/fx/fy/cx/cy`)
  - `camera` (`type/radius/center/noise/lookat/up`)

---

## 9. Notes on Partial Point Clouds
- Saved `cam_ex_XX.npy` is camera-to-world transform in current implementation.
- `partial_pc_XX.npy` is reconstructed from rendered depth + intrinsics + extrinsics in world frame.
- `partial_pc_cam_XX.npy` stores the same sampled points in camera frame.
- Point cloud geometry depends on both intrinsics and extrinsics.

---

## 10. MJWarp Sampling Tuning Notes
- If the goal is to collect about `100` valid grasps per object-scale quickly, keep `output.max_cap = 100` and prefer `run_mjw.py` batch sizes near that target, typically `128`, `256`, or `512`.
- Do not assume larger is faster for small-cap collection. A very large `batch_size` such as `4096` makes each MJWarp step heavier, so the extforce stage can become slower even though GPU occupancy is higher.
- `batch_size = 4096` is better suited to large-scale grasp harvesting, not to quickly stopping after about `100` valid grasps.
- Smaller `batch_size` also means fewer candidates are pushed through collision and `sim_grasp`, so surface sampling density should be reduced accordingly.
- For the current `max_cap = 100` workflow, `sampling.n_points = 1024` is the recommended default. This avoids waiting too long in collision filtering and `sim_grasp` while still producing enough candidates for extforce.

---

## 11. Quick Commands
```bash
# 1) Grasp generation for all object-scales
python run_multi.py -c configs/run_YCB_liberhand.json -j 4

# 2) Render partial point clouds after grasp generation
python run_warp_render.py -c configs/run_YCB_liberhand.json -j 1

# 3) Visualize one object-scale partial point clouds
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0 --show-cam-frames
```

---

## 12. CPU Benchmark Tips
Use these only when you want cleaner CPU benchmarking for `run_multi.py`.

Clear Linux page cache before a benchmark run (requires `sudo`):

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
```

Pin `run_multi.py` and all of its child processes to a CPU set with `taskset`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0-15 python run_multi.py -c configs/run_YCB_liberhand.json -j 16
```

On a NUMA machine, prefer `numactl` so CPU affinity and memory locality stay together:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
numactl --physcpubind=0-15 --localalloc \
python run_multi.py -c configs/run_YCB_liberhand.json -j 16
```

Notes:
- `taskset` affinity is inherited by child processes started by `run_multi.py`.
- Setting BLAS/OpenMP thread counts to `1` avoids oversubscription when `run_multi.py` already uses process-level parallelism.
- Dropping page cache is mainly for reproducible benchmarking, not a normal production requirement.

---

## 13. Testing
```bash
pytest -q
```
