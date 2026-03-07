# AGENTS.md

## Repository Purpose
MuJoCo-based dexterous grasp sampling for 3D objects.
Mainline focus is offline grasp configuration generation, not superquadric fitting.

## Planning Index
- Active plan: `TODO.md`

## Mainline Code (Do Not Drift)
- `run_multi.py`
- `run.py`
- `src/mj_ho.py` (core, keep stable)
- `src/sample.py`
- `src/dataset_objects.py`

## Architecture Notes
- Main data flow: object mesh -> surface points/normals -> grasp frame sampling -> MuJoCo collision/stability filtering -> HDF5 grasp states.
- Visualization scripts are intentionally isolated under `tools/visualization/` and are not part of the mainline pipeline.
- SQ/Superquadric and SDF generation are removed from mainline.

## Dataset Interface
- Preferred dataset root is `assets/objects/processed`.
- This repo uses a symlink interface to external mesh repository:
  - `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`
- Override dataset root via env:
  - `DEXGRASP_OBJECTS_ROOT=/abs/path/to/processed_or_object_root`
- Default merged dataset list in mainline:
  - `["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`
- `DatasetObjects` exposes a global integer id space over the merged list; `run.py` selects objects by global id.

## Config Policy (Mandatory)
- Mainline is config-first: all CLI entrypoints must load a JSON config.
- Do not rebuild defaults inside Python code (`build_default_*` style is disallowed).
- Default entry config for scripts is:
  - `configs/run_YCB_liberhand.json`
- Config set naming:
  - `<dataset_group>_<hand>.json`, where dataset group in `{DGN2, YCB, HOPE}` and hand in `{liberhand, inspire, liberhand2}`.
- `DGN2` means merged datasets:
  - `["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`
- Invalid/missing config fields should fail fast with explicit error.

## Output Contract
Main output HDF5 fields:
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`

## Unified Dataset Format (Summary)
- Internal sample representation should include:
  - `object_id`, `dataset`, `scale`, `hand_name`
  - hand pose as `wxyz + qpos` relative-to-object convention
  - object point cloud `(K,3)` and source metadata
- Public references for format comparison are tracked in:
  - `docs/dataset_format_and_scale.md`

## Scale Policy (Summary)
- `YCB/HOPE/MSO`: keep real-world scale (`scale=1.0`).
- `ShapeNetCore/ShapeNetSem`: enable scale by dataset policy.
- Current mainline decision: per-object fixed random scale (deterministic by object id + seed), not per-grasp random scale.
- Manifest gating must happen first: only `manifest.process_meshes.json` entries with `process_status=success` are eligible.

## Collaboration Rules
- Keep edits focused on requested tasks.
- Avoid structural rewrites of `src/mj_ho.py` unless strictly necessary.
- Validate each refactor item with explicit completion checks in `TODO.md`.

## Global GitHub-KB Path (Mandatory)
- `github-kb` default path is **global only**: `/home/ccs/github-kb`.
- Do **not** clone or maintain `github-kb` inside this repository.
- Any repo exploration/knowledge-base update must target `/home/ccs/github-kb/CLAUDE.md`.

## Git Remote Reminder
User note: remote is managed with:
`git remote set-url origin git@github.com:HomeworldL/`

Ensure final remote URL targets the real repository SSH path, e.g.
`git@github.com:HomeworldL/dexgrasp_sample.git`.
