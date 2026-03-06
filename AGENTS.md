# AGENTS.md

## Repository Purpose
This repository builds a MuJoCo-based grasp sampling pipeline for five-finger dexterous hands.
It processes object 3D assets (mainly YCB-format meshes/XML), samples candidate grasp poses from object surface point clouds, validates them in simulation, and saves successful grasps as datasets for downstream training/evaluation.

## Core Capabilities
- Object dataset indexing and loading (`src/dataset_objects.py`)
- Surface point cloud sampling + normal extraction from object meshes
- Candidate grasp frame generation (`src/sample.py`)
- Hand-object simulation, contact checking, closing, and stability validation under external force (`src/mj_ho.py`)
- Optional superquadric/EMS utilities for geometric approximation (`src/EMS/`, `src/sq_handler.py`)
- Multi-object batch sampling (`run_multi.py`)

## Main Entrypoints
- `run.py`: Grasp sampling for `liberhand_right`
- `run_liberhand2.py`: Grasp sampling for `liberhand2_right`
- `run_multi.py`: Parallel execution across objects in `assets/ycb_datasets`

## Typical Workflow
1. Load target object meshes and metadata from `assets/ycb_datasets/<object_name>/`
2. Sample object point cloud + normals
3. Generate grasp frame candidates around points
4. Convert candidates into hand root pose + finger preset joint vectors
5. Reject immediate collisions (`prepared`, `approach`, `init` checks)
6. Simulate grasp closing and keep candidates with enough contacts
7. Validate force robustness in a second MuJoCo instance
8. Save valid trajectories/states to `outputs/<hand_name>/<object_name>/grasp_data.h5`

## Output Data Contract
The main output HDF5 includes:
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`

Each row is one valid grasp sample in MuJoCo qpos layout for the selected hand model.

## Important Directories
- `assets/hands/`: dexterous hand XML and kinematic config
- `assets/ycb_datasets/`: object meshes/XML/URDF/metadata
- `src/`: core pipeline modules
- `outputs/`: generated grasp datasets
- `logs/`: per-object run logs from parallel jobs

## Environment Notes
This repo expects a Python environment with at least:
- `numpy`, `scipy`, `torch`, `trimesh`, `open3d`, `mujoco`, `mujoco-viewer`, `h5py`, `tqdm`, `transforms3d`, `viser`

MuJoCo runtime must be correctly installed/licensed for local simulation.

## Collaboration Rules for AI Agents
- Keep changes minimal and task-focused; avoid refactoring unrelated modules.
- Do not delete or rewrite generated outputs unless explicitly asked.
- Prefer documenting assumptions when touching grasp conventions (pose frame, quaternion order, joint layout).
- Validate any pipeline edits by running a small single-object sample first.

## Git Remote Reminder
User note: remote is managed with:
`git remote set-url origin git@github.com:HomeworldL/`

When pushing, ensure the final `origin` URL points to the intended SSH repo path (typically `git@github.com:HomeworldL/<repo>.git`).
