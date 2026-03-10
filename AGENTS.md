# AGENTS.md

## Repository Purpose
MuJoCo-based dexterous grasp sampling for 3D objects.
Mainline focus is offline grasp configuration generation.

## Planning Index
- Active plan: `TODO.md`

## TODO Lifecycle (Mandatory)
- Keep only one active plan file at repo root: `TODO.md`.
- Before replacing an active TODO, archive it to `docs/` using:
  - `docs/TODO_<timestamp>_history.md`
- When all items in the current `TODO.md` are checked:
  - archive that TODO to `docs/` with the same naming rule
  - create a new root `TODO.md` for the next iteration
- During execution, always read the active root `TODO.md` first; history files are reference only.

## Mainline Code (Do Not Drift)
- `run_multi.py`
- `run.py`
- `run_warp_render.py`
- `src/mj_ho.py` (core, keep stable)
- `src/sample.py`
- `src/dataset_objects.py`

## Architecture Notes
- Main data flow: object mesh -> surface points/normals -> grasp frame sampling -> MuJoCo collision/stability filtering -> HDF5 grasp states.
- Post-sampling vision flow: after grasp sampling for each object-scale, run `run_warp_render.py` to render multi-view partial point clouds from scaled `coacd.obj`.

## Dataset Interface
- Preferred dataset root is `assets/objects/processed`.
- This repo uses a symlink interface to external mesh repository:
  - `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`
- Default merged dataset list in mainline:
  - `["YCB"]`
- `DatasetObjects` exposes a global integer id space over the merged list; `run.py` selects objects by global id.

## Scale Policy
- All datasets use unified fixed scale list from config (`dataset.scales`).
- Current mainline default list for liberhand configs:
  - `[0.08, 0.10, 0.12, 0.14, 0.16]`
- Scale granularity is object-scale flattened indexing (one entry per object per scale).
- Manifest gating must happen first: only `manifest.process_meshes.json` entries with `process_status=success` are eligible.

## Sampling Pipeline (End-to-End)
- Set deterministic seed for `numpy/random/torch` (including CUDA and cuDNN deterministic flags).
- Load object metadata and meshes via `DatasetObjects`, and sample object point cloud + normals (poisson).
- Sample candidate grasp frames from point cloud/normals (`sample_grasp_frames`), then convert frame pose to hand pose convention:
  - apply grasp-to-palm rotation alignment
  - convert rotation to quaternion and store as `wxyz`
  - build base hand pose `[tx,ty,tz,qw,qx,qy,qz]`
- Build three pre-grasp states per candidate:
  - `qpos_prepared`: base pose + prepared finger joints
  - `qpos_approach`: same pose + approach finger joints
  - `qpos_init`: approach joints with a local negative-z offset transformed to world frame
- Run staged collision filtering in MuJoCo (`MjHO`) for each candidate:
  - reject if `prepared` collides
  - reject if `approach` collides
  - reject if `init` collides
- For collision-free candidates:
  - simulate closing grasp to get `qpos_grasp`
  - require sufficient hand-object contacts (>=4)
  - run external-force stability validation in a second simulator (`object_fixed=False`)
  - keep only validated grasps
- Persist outputs as HDF5 first:
  - datasets: `qpos_init`, `qpos_approach`, `qpos_prepared`, `qpos_grasp`
  - use preallocated capacity + runtime truncate to final valid size
  - periodic flush/GC during long runs
- After HDF5 is finalized, load the same arrays and convert to `grasp.npy` (values must be identical to HDF5).
- After grasp outputs are ready, render object partial point clouds with Warp (`run_warp_render.py`) and save under:
  - `datasets/<config_stem>/<object>/scaleXXX/<warp_render.output_subdir>/`
  - default subdir: `partial_pc_warp`
- Point cloud is saved separately and is not bundled into `grasp.npy`.

## Config Policy (Mandatory)
- Mainline is config-first: all CLI entrypoints must load a JSON config.
- Do not rebuild defaults inside Python code (`build_default_*` style is disallowed).
- Default entry config for scripts is:
  - `configs/run_YCB_liberhand.json`
- Config set naming:
  - `<dataset_group>_<hand>.json`, where dataset group in `{YCB, DGN, DGN2, HOPE}` and hand in `{liberhand, inspire, liberhand2}`.
- `DGN2` means merged datasets:
  - `["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`
- Invalid/missing config fields should fail fast with explicit error.

## Unified Dataset Format (Summary)
- Internal sample representation should include:
  - one output pair per object-scale: first `grasp.h5`, then convert to `grasp.npy`
  - `grasp.npy` must be converted from `grasp.h5` with identical stored grasp values
  - `grasp.h5` sample schema:
    - `object_id: str`
    - `object_name: str`
    - `dataset: str`
    - `scale: float`
    - `hand_name: str`
    - `rot_repr: "wxyz+qpos"`
    - `qpos_init: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_approach: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_prepared: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_grasp: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `meta: {}`
  - point cloud is stored separately and must not be bundled into `grasp.npy`
  - partial point cloud rendering output (post-grasp, separate from grasp arrays):
    - `cam_in.npy`
    - `cam_ex_XX.npy`
    - `partial_pc_XX.npy`
- Public references for format comparison are tracked in:
  - `docs/dataset_format_and_scale.md`

## Collaboration Rules
- Keep edits focused on requested tasks.
- Avoid structural rewrites of `src/mj_ho.py` unless strictly necessary.

## Python Code Style (Concise)
- Follow PEP 8 and format with `black` + `isort`.
- Add type hints for public functions and key data structures.
- Keep entry scripts thin; place core logic in `src/`.
- Config-first: do not hide defaults in code when config is required.
- Use `logging` instead of ad-hoc `print` for runtime logs.
- Fail fast with explicit, contextual error messages.
- Add/maintain minimal tests for core pipeline and bug regressions.
- Keep commits small and focused with clear messages.

## Git Remote Reminder
User note: remote is managed with:
`git remote set-url origin git@github.com:HomeworldL/`

Ensure final remote URL targets the real repository SSH path, e.g.
`git@github.com:HomeworldL/dexgrasp_sample.git`.
