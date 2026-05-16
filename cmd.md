# cmd.md

仓库根目录下新版脚本命令手册。以下示例都默认在仓库根目录执行。

## 新版路径约定

新版 `objdata_*` 目录中：

- `meshes/`：raw shared mesh
- `meshes_normalized/`：normalized shared mesh
- `scaleXXX/`、`native/`：只保留 `object.xml`、`object.urdf`、`pc_warp/*`、`object.usd`、`configuration/*`

对命令最重要的影响是：

- CPU 主链 `run.py` 仍然直接吃 `scaleXXX/object.xml` 和 `scaleXXX/pc_warp/*`
- 需要 `coacd.obj` 的脚本，现在对 `scaleXXX` 应传对象级共享路径：
  - `datasets/objdata_*/<object>/meshes_normalized/coacd.obj`
- `native` 资产若需要 `coacd.obj`，应传：
  - `datasets/objdata_*/<object>/meshes/coacd.obj`

## 通用变量

```bash
CFG_RUN=configs/run_YCB_liberhand_right.json
CFG_ASSET=configs/assets_YCB.json
OBJ_NAME=YCB_002_master_chef_can
SCALE_TAG=scale080
OBJ_KEY=${OBJ_NAME}__${SCALE_TAG}

OBJ_ASSET_DIR=datasets/objdata_YCB/${OBJ_NAME}/${SCALE_TAG}
OBJ_COACD_NORM=datasets/objdata_YCB/${OBJ_NAME}/meshes_normalized/coacd.obj
OBJ_GRASP_DIR=datasets/graspdata_YCB_liberhand_right/${OBJ_NAME}/${SCALE_TAG}
```

切到 DGN 时，主要替换：

```bash
CFG_RUN=configs/run_DGN_liberhand_right.json
CFG_ASSET=configs/assets_DGN.json
```

如果本地没有对应 `run_DGN_*` 配置，就按你自己的 DGN run config 替换。

## 资产准备

```bash
python prepare_object_assets.py -c ${CFG_ASSET} --force --jobs 8
python prepare_object_usds.py -c ${CFG_ASSET}
python print_dataset_objects.py -c ${CFG_RUN}
python print_dataset_objects.py -c ${CFG_RUN} --key-substr master_chef_can
python print_dataset_objects.py -c ${CFG_RUN} --object-name ${OBJ_NAME}
```

## CPU 采样主链

```bash
python run.py -c ${CFG_RUN} \
  --object-scale-key ${OBJ_KEY} \
  --mjcf-path ${OBJ_ASSET_DIR}/object.xml \
  --global-pc-path ${OBJ_ASSET_DIR}/pc_warp/global_pc.npy \
  --global-normals-path ${OBJ_ASSET_DIR}/pc_warp/global_normals.npy \
  --output-dir ${OBJ_GRASP_DIR}

python run_multi.py -c ${CFG_RUN} -j 16 --force
python build_dataset_splits.py -c ${CFG_RUN}
python sim_dataset.py -c ${CFG_RUN} --split test --dtype float32 -v
python sim_dataset.py -c ${CFG_RUN} --split test --dtype float64 -v
```

## GPU / MJWarp

```bash
python run_mjw.py -c ${CFG_RUN} \
  --object-scale-key ${OBJ_KEY} \
  --coacd-path ${OBJ_COACD_NORM} \
  --mjcf-path ${OBJ_ASSET_DIR}/object.xml \
  --output-dir ${OBJ_GRASP_DIR} \
  --batch-size 256 \
  --device cuda:0

python run_multi_mjw.py -c ${CFG_RUN} -j 4 --batch-size 256 --device cuda:0 --force

python run_mjw_no_save.py -c ${CFG_RUN} \
  -k ${OBJ_KEY} \
  --batch-size 32 \
  --max-candidates 256 \
  --n-points 1024 \
  --device cuda:0

python run_collision.py -c ${CFG_RUN} \
  -k ${OBJ_KEY} \
  --batch-size 32 \
  --max-candidates 256 \
  --n-points 1024 \
  --device cuda:0 -v
```

## Warp 渲染

`run_warp_render.py` 读取的是 asset config，不是 run config。

```bash
python run_warp_render.py -c ${CFG_ASSET} -j 2
python run_warp_render.py -c ${CFG_ASSET} -k ${OBJ_KEY} -j 1 --gpu-lst 0
python run_warp_render.py -c ${CFG_ASSET} -i 0 -j 1 --gpu-lst 0 --force
```

## 可视化

```bash
python vis_obj.py -c ${CFG_RUN} -k ${OBJ_KEY}
python vis_obj_scale.py -c ${CFG_RUN} -k ${OBJ_KEY}
python vis_ho.py -c ${CFG_RUN} -k ${OBJ_KEY}
python vis_grasp.py -c ${CFG_RUN} -k ${OBJ_KEY}
python vis_grasp.py -c ${CFG_RUN} -k ${OBJ_KEY} --vis-ids 0,10,-1 --frame-stage qpos_grasp
python vis_pc.py -c ${CFG_RUN} -k ${OBJ_KEY} --show-cam-frames
python vis_grasp_mujoco.py -c ${CFG_RUN} --object-scale-dir ${OBJ_ASSET_DIR} --dtype float32 -v
```

`vis_grasp_mujoco.py` 现在既支持传 `objdata` 里的 asset dir，也支持直接传 `graspdata` 里的输出目录。

## Shape Cluster / RL 元数据

```bash
python run_shape_cluster.py -c ${CFG_ASSET} --force
python build_dataset_splits_rl.py -c ${CFG_ASSET} --force
python vis_shape_cluster.py -c ${CFG_ASSET}
```

## Demo / Debug

```bash
python demo.py -c ${CFG_RUN} \
  --object-scale-key ${OBJ_KEY} \
  --coacd-path ${OBJ_COACD_NORM} \
  --mjcf-path ${OBJ_ASSET_DIR}/object.xml -v

python demo_grasp.py -c ${CFG_RUN} \
  --object-scale-key ${OBJ_KEY} \
  --coacd-path ${OBJ_COACD_NORM} \
  --mjcf-path ${OBJ_ASSET_DIR}/object.xml -v

python demo_usd.py --asset-dir ${OBJ_ASSET_DIR} --headless

python debug_viser_grasp.py -c ${CFG_RUN} \
  --object-scale-key ${OBJ_KEY} \
  --coacd-path ${OBJ_COACD_NORM} \
  --mjcf-path ${OBJ_ASSET_DIR}/object.xml \
  --asset-dir ${OBJ_ASSET_DIR} \
  --skip-candidates 0 \
  --max-candidates 1 -v
```

## 脚本目录 `scripts/`

这些是实验 / sweep / 清理工具，不纳入主线数据流程，但可以直接这样用：

```bash
python scripts/delete_pc_warp_dirs.py --dataset-dir datasets/graspdata_YCB_liberhand_right --subdir pc_warp
python scripts/delete_pc_warp_dirs.py --dataset-dir datasets/graspdata_YCB_liberhand_right --subdir pc_warp --execute

python scripts/summarize_grasp_logs.py --log-dirs logs/run/graspdata_YCB_liberhand_right

python scripts/friction_only_sweep.py --help
python scripts/inspire_sim_grasp_sweep.py --help
python scripts/liberhand_actuation_sweep.py --help
python scripts/sim_grasp_param_sweep.py --help
python scripts/solimp_only_sweep.py --help
python scripts/solimp_solref_experiment.py --help
python scripts/solref_only_sweep.py --help
```

`scripts/sweep_utils.py` 是辅助模块，不单独作为入口脚本执行。

## 先看帮助

如果想先确认参数面，直接跑：

```bash
python run.py --help
python run_multi.py --help
python run_mjw.py --help
python run_warp_render.py --help
python build_dataset_splits.py --help
python sim_dataset.py --help
```
