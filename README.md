# dexgrasp_sample

MuJoCo-based dexterous grasp sampling pipeline for object 3D assets.

## Mainline
- `run.py`
- `run_multi.py`
- `src/mj_ho.py`
- `src/sample.py`
- `src/dataset_objects.py`

## What It Does
1. Load object assets from a dataset root
2. Sample surface points and normals
3. Generate grasp-frame candidates
4. Run MuJoCo hand-object filtering and grasp simulation
5. Validate grasp stability and export valid states

## Dataset Root (External Interface)
This project now supports generic dataset layouts through `DatasetObjects`.

Default root priority:
1. `DEXGRASP_OBJECTS_ROOT`
2. `assets/objects/processed`

`assets/objects` is expected to be a symlink interface (no large dataset copy).

## Config-First Entrypoints
All CLI entrypoints are config-driven and require a JSON config schema:
- `run.py`
- `run_multi.py`
- `vis_obj.py`
- `vis_ho.py`
- `run_liberhand2.py`

Default config for all entrypoints:
- `configs/run_YCB_liberhand.json`

Available config matrix:
- `run_DGN2_liberhand.json`, `run_YCB_liberhand.json`, `run_HOPE_liberhand.json`
- `run_DGN2_liberhand2.json`, `run_YCB_liberhand2.json`, `run_HOPE_liberhand2.json`
- `run_DGN2_inspire.json`, `run_YCB_inspire.json`, `run_HOPE_inspire.json`

Dataset semantics:
- `DGN2 = ["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`

If config fields are missing/invalid, program exits with explicit error (no in-code default config build).

## Visualization Scripts
Visualization and plotting tools are separated from mainline under:
- `tools/visualization/`

## Quick Start
```bash
python run.py -c configs/run_YCB_liberhand.json -o 002_master_chef_can
python run_multi.py -c configs/run_DGN2_liberhand.json -j 4 --script run.py
python vis_obj.py -c configs/run_YCB_liberhand.json -i 0
python vis_ho.py -c configs/run_YCB_liberhand2.json -i 0
```

## Tests
```bash
pytest -q
```

## Dependencies
See `requirements.txt`.
