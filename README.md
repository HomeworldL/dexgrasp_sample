# dexgrasp_sample

MuJoCo-based grasp sampling pipeline for 3D objects and multiple five-finger dexterous hands.

## What This Repo Does
- Loads processed object assets (YCB-style meshes + XML/URDF)
- Samples grasp candidates from object surface points and normals
- Simulates hand-object interaction in MuJoCo
- Filters candidates by collision, contact richness, and external-force stability
- Exports valid grasp states as HDF5 datasets

This project is designed for building grasp datasets rather than online control.

## Supported Hand Models
- `assets/hands/liberhand/liberhand_right.xml`
- `assets/hands/liberhand2/liberhand2_right.xml`

Main scripts:
- `run.py` -> `liberhand_right`
- `run_liberhand2.py` -> `liberhand2_right`

## Repository Structure
- `src/dataset_objects.py`: object dataset indexing, mesh/point-cloud loading
- `src/sample.py`: grasp frame candidate generation + FPS downsampling
- `src/mj_ho.py`: MuJoCo hand-object environment and grasp validation logic
- `src/sq_handler.py`, `src/EMS/`: superquadric utilities
- `run.py`, `run_liberhand2.py`: single-object grasp sampling
- `run_multi.py`: parallel multi-object execution
- `assets/`: hands and YCB object assets
- `outputs/`: generated grasp data

## Pipeline Overview
1. Choose target object (`-o <object_name>`)
2. Sample object point cloud and normals
3. Generate candidate grasp frames around surface points
4. Convert candidate transforms to hand root pose + finger joint preset
5. Collision checks at `prepared`, `approach`, `init`
6. Simulate closing to get `qpos_grasp`
7. Keep grasps with enough hand-object contacts
8. Validate stability under external force perturbation
9. Save valid samples to HDF5

## Installation
No lockfile is currently provided. Create your own Python environment and install dependencies:

```bash
pip install numpy scipy torch trimesh open3d mujoco mujoco-viewer h5py tqdm transforms3d viser
```

## Quick Start
Run one object with LiberHand:

```bash
python run.py -o 002_master_chef_can
```

Run one object with LiberHand2:

```bash
python run_liberhand2.py -o 006_mustard_bottle
```

Run parallel jobs across objects:

```bash
python run_multi.py -j 4 --script run.py
```

## Output Format
For each `(hand, object)` pair, outputs are written to:

```text
outputs/<hand_name>/<object_name>/grasp_data.h5
```

Datasets inside `grasp_data.h5`:
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`

Each row corresponds to one valid grasp sample in MuJoCo qpos convention.

## Notes
- This codebase currently focuses on data generation and analysis scripts.
- Some modules under `old/` and `src/backup/` are historical and not part of the main pipeline.
- If MuJoCo rendering fails, verify local MuJoCo/OpenGL setup first.
