# AGENTS.md

## Repository Purpose
MuJoCo-based dexterous grasp sampling for 3D objects.
Mainline work focuses on offline grasp configuration generation.

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
- This repo uses a symlink interface to an external mesh repository:
  - `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`
- Default merged dataset list in mainline:
  - `["YCB"]`
- `DatasetObjects` exposes a global integer id space over the merged list; `run.py` selects objects by global id.

## Scale Policy
- All datasets use the unified fixed scale list from config (`dataset.scales`).
- Current mainline default list for liberhand configs:
  - `[0.08, 0.10, 0.12, 0.14, 0.16]`
- Scale granularity is object-scale flattened indexing (one entry per object per scale).
- Manifest gating must happen first: only `manifest.process_meshes.json` entries with `process_status=success` are eligible.

## Sampling Pipeline (CPU Version)
- Set deterministic seeds for `numpy/random/torch/Open3D` (including CUDA and cuDNN deterministic flags).
- Load object metadata and meshes via `DatasetObjects`, and sample object point cloud + normals (poisson).
- Sample candidate grasp frames from point cloud/normals (`sample_grasp_frames`), then convert frame pose to the hand pose convention:
  - apply grasp-to-palm rotation alignment from `hand.transform`
  - convert rotation to quaternion and store as `wxyz`
  - build base hand pose `[tx,ty,tz,qw,qx,qy,qz]`
- Build three pre-grasp states per candidate:
  - `qpos_prepared`: base pose + prepared finger joints
  - `qpos_approach`: same base pose + approach finger joints
  - `qpos_init`: approach joints with a local negative-z offset transformed to world coordinates
- Run staged collision filtering in MuJoCo (`MjHO`) for each candidate:
  - reject if `prepared` collides
  - reject if `approach` collides
  - reject if `init` collides
- For collision-free candidates:
  - simulate closing grasp to get `qpos_grasp`
  - require sufficient hand-object contacts (`sim_grasp.contact_min_count`, current default `>=4`)
  - build `qpos_squeeze` from `qpos_grasp` using `extforce.grip_delta`
  - run external-force stability validation in a second simulator (`object_fixed=False`) with two stages:
    - rebuild a validation pregrasp from `qpos_squeeze` pose plus the saved prepared joints
    - close from rebuilt `qpos_prepared` toward `qpos_squeeze`, then gate no-force settling drift
    - then apply six-direction external forces using the settled pose as the baseline
  - keep only validated grasps
- Persist outputs as HDF5 first (`grasp.h5`):
  - datasets: `qpos_init`, `qpos_approach`, `qpos_prepared`, `qpos_grasp`, `qpos_squeeze`
  - mainline stored dtype for grasp arrays is `float32`
  - use preallocated capacity + runtime truncate to the final valid size
  - periodic flush/GC during long runs
- Persist failure samples separately as `grasp_fail.h5` and `grasp_fail.npy`:
  - fields: `qpos_fail`, `failure_stage`
  - current retained stages are `prepared_contact`, `insufficient_contact`, and `extforce_failure`
  - `approach_contact` and `init_contact` are not exported into the failure dataset
  - `qpos_fail` stores the stage-specific failed state:
    - `prepared_contact` stores the colliding prepared state
    - `insufficient_contact` stores the post-close `qpos_grasp`
    - `extforce_failure` stores the failed `qpos_squeeze`
  - if `valid_count < output.min_valid_count`, both positive and failure files must be truncated to zero rows
  - otherwise failure samples are deterministically shuffled with config seed and truncated to `floor(output.fail_keep_ratio * valid_count)`
- After `grasp.h5` is finalized, load the same arrays and convert them to `grasp.npy` (values must be identical to HDF5).
- `sim_dataset.py` is the dataset-level replay/validation entrypoint for `train.json` / `test.json`.
  - it keeps the stored `qpos_init` pre-check
  - it rebuilds replay `qpos_prepared` from stored `qpos_squeeze` pose plus saved prepared joints
  - it validates stored `qpos_squeeze` with `sim_under_extforce(qpos_target, rebuilt_qpos_prepared, ...)`
  - support `--dtype float32` or `--dtype float64` to compare precision-sensitive extforce success rates
- After grasp outputs are ready, render object partial point clouds with Warp and save under:
  - `datasets/<dataset_tag>/<object>/scaleXXX/<warp_render.output_subdir>/`
  - `dataset_tag` rule: replace config stem prefix `run_` with `graspdata_`
  - default subdir: `pc_warp`
  - additionally export `global_pc.npy` under the same `pc_warp/` directory
  - `global_pc.npy` is a separate object-level point cloud, not a merge of rendered partial views
  - current mainline default is world-frame `float32` points with shape `(4096, 3)`, sampled directly from `coacd.obj`
- Point cloud is saved separately and must not be bundled into `grasp.npy`.

## Sampling Pipeline (GPU Version)
- Use MJWarp.
- For `run_mjw.py`, keep `output.max_cap=100` and cap extforce wall-clock time with `output.max_time_sec` (current default: `180s`).
- If the goal is to collect about `100` valid grasps per object-scale quickly, prefer `batch_size` near `max_cap`, typically `128`, `256`, or `512`.
- Do not default to overly large `batch_size` such as `4096` for small-cap collection: one MJWarp step becomes too heavy and the extforce stage slows down noticeably.
- `batch_size=4096` is more appropriate for large-scale grasp harvesting, not for quickly reaching `max_cap=100`.
- Because reducing `batch_size` also reduces total sampled candidates flowing into later stages, `sampling.n_points` should also be moderated.
- For the current `max_cap=100` target, `sampling.n_points=2048` is the current default; it keeps candidate coverage reasonable without making collision filtering and `sim_grasp` excessively slow.

## Dataset Split Policy
- Use `build_dataset_splits.py` to scan completed grasp/render outputs and write `datasets/<dataset_tag>/train.json` and `datasets/<dataset_tag>/test.json`.
- Split by `object_name`, not by object-scale entry; all scales of the same object must stay in the same split to avoid leakage.
- Default split ratio is approximately 80/20 over unique objects, shuffled with the config seed.
- A split record is valid only when positive grasp outputs, failure grasp outputs, and the required render outputs all exist.
- The split record must include:
  - `grasp_h5_path`, `grasp_npy_path`
  - `grasp_h5_fail_path`, `grasp_fail_npy_path`
  - `global_pc_path`
- Empty grasp files must be filtered out before the final manifests are used.
- Manifest paths should be stored relative to the dataset root so the dataset can be moved without rewriting absolute paths.

## Config Policy (Mandatory)
- Mainline is config-first: all CLI entrypoints must load a JSON config.
- Do not rebuild defaults inside Python code (`build_default_*` style is disallowed).
- Default entry config for scripts is:
  - `configs/run_YCB_liberhand.json`
- Current mainline config grouping order is:
  - `seed`, `dataset`, `hand`, `sampling`, `sim_grasp`, `extforce`, `output`, `warp_render`
- Config set naming:
  - `<dataset_group>_<hand>.json`, where dataset group is in `{YCB, DGN, DGN2, HOPE}` and hand is in `{liberhand, inspire, liberhand2}`.
- `DGN2` means merged datasets:
  - `["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`
- Invalid or missing config fields should fail fast with explicit errors.

## Unified Dataset Format (Summary)
- Internal sample representation should include:
  - one required output per object-scale: `grasp.h5`
  - one required derived output per object-scale: `grasp.npy`
  - one required failure output per object-scale: `grasp_fail.h5`
  - one required derived failure output per object-scale: `grasp_fail.npy`
  - `grasp.npy` must be converted from `grasp.h5` with identical stored grasp values
  - `grasp.h5` sample schema:
    - `object_name: str`
    - `scale: float`
    - `hand_name: str`
    - `rot_repr: "wxyz+qpos"`
    - `qpos_init: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_approach: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_prepared: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_grasp: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_squeeze: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `meta: {}`
  - `grasp_fail.h5` sample schema:
    - `object_name: str`
    - `scale: float`
    - `hand_name: str`
    - `rot_repr: "wxyz+qpos"`
    - `qpos_fail: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `failure_stage: str`
  - stored `qpos_prepared` remains the original candidate pregrasp state
  - replay/extforce validation rebuilds a pregrasp using `qpos_squeeze` pose plus stored prepared joints; it does not directly replay the saved `qpos_prepared` pose
  - point cloud is stored separately and must not be bundled into `grasp.npy`
  - partial point cloud rendering output (post-processing, separate from grasp arrays):
    - `global_pc.npy`
    - `cam_in.npy`
    - `cam_ex_XX.npy`
    - `partial_pc_XX.npy`
    - `partial_pc_cam_XX.npy`

## Collaboration Rules
- Keep edits focused on the requested task.
- Avoid structural rewrites of `src/mj_ho.py` unless strictly necessary.

## Planning Index
- Active plan: `TODO.md`

## Bilingual Sync (Mandatory)
- `AGENTS_zh.md` is the Chinese translation of this file and must stay aligned with `AGENTS.md`.
- Every time `AGENTS.md` is updated, update `AGENTS_zh.md` in the same change.

## Docs Naming (Mandatory)
- Self-maintained docs under `docs/` must use:
  - `YYYYMMDD_<类别>_<标题概括>.md`
- Required category labels:
  - `TODO`
  - `方案`
  - `调研`
- Keep the date first, then category, then a short summary title.
- `docs/` is reference-only in this repo workflow and should not be uploaded to git; keep docs ignored and local unless the user explicitly requests otherwise.
- Exported scan/data artifacts (for example `github_dexterous_grasp_scan.*`) do not need to follow this rule.

## TODO Lifecycle (Mandatory)
- Keep only one active plan file at repo root: `TODO.md`.
- Before replacing an active TODO, archive it to `docs/` using:
  - `docs/YYYYMMDD_TODO_<标题概括>_history.md`
- When all items in the current `TODO.md` are checked:
  - archive that TODO to `docs/` with the same naming rule
  - create a new root `TODO.md` for the next iteration
- During execution, always read the active root `TODO.md` first; history files are reference only.

## Python Code Style (Concise)
- Follow PEP 8 and format with `black` + `isort`.
- Add type hints for public functions and key data structures.
- Keep entry scripts thin; place core logic in `src/`.
- Config-first: do not hide defaults in code when config is required.
- Use `logging` instead of ad-hoc `print` for runtime logs.
- Fail fast with explicit, contextual error messages.
- For straightforward logic, prefer direct control flow; avoid unnecessary fallback branches, compatibility branches, and broad `try/except` blocks.
- Add/maintain minimal tests for core pipeline and bug regressions.
- Keep commits small and focused with clear messages.

## Git Remote Reminder
User note: remote is managed with:
`git remote set-url origin git@github.com:HomeworldL/`

Ensure the final remote URL targets the real repository SSH path, for example:
`git@github.com:HomeworldL/dexgrasp_sample.git`.
