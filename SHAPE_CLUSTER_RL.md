# Shape Cluster RL Quick Reference

## Run
Prepare object assets first:
```bash
python prepare_object_assets.py -c configs/assets_YCB.json --force --jobs 8
```

Run shape clustering:
```bash
python run_shape_cluster.py -c configs/assets_YCB.json --force
```

Build RL split:
```bash
python build_dataset_splits_rl.py -c configs/assets_YCB.json --force
```

Render cluster thumbnails:
```bash
python vis_shape_cluster.py -c configs/assets_YCB.json
```

## What It Uses
Shape clustering uses:
- `datasets/objdata_YCB/<object>/scale120/pc_warp/global_pc.npy`
- centered + unit-radius normalized point clouds
- object-level AE feature extraction
- KMeans over object features

Shape clustering does not use:
- `native`
- `grasp.h5`
- `grasp.npy`
- `run_warp_render.py` partial point clouds

## Main Outputs
Shape cluster directory:
- `datasets/objdata_YCB/_meta/shape_cluster/v1_ae128_k4_seed0/`

Main files:
- `meta.json`: config and file index
- `object_features.npy`: object feature vectors
- `cluster_centers.npy`: KMeans centers
- `object_cluster.json`: object to cluster mapping
- `cluster_index.json`: cluster to ordered member mapping
- `curriculum_index.json`: RL curriculum order from center outward
- `train_history.json`: AE training loss history
- `ae_state_dict.pt`: trained AE checkpoint

RL split directory:
- `datasets/objdata_YCB/_meta/rl_split/v1_ae128_k4_seed0_test20_seed0/`

Main files:
- `train_rl.json`
- `test_rl.json`
- `train_cluster_index.json`
- `test_cluster_index.json`
- `meta.json`

## Meaning
- clustering is object-level
- RL records are object-scale-level
- all scales of one object inherit the same cluster metadata
- members inside each cluster are sorted by distance to the cluster center object
