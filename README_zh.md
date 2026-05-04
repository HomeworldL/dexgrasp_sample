# dexgrasp_sample

一个基于 MuJoCo 的离线灵巧手抓取采样仓库，当前面向“物体资产准备 -> 抓取状态构造 -> 可选聚类与 RL 元数据”的整套数据流程。

当前主线流程是：

1. 先准备物体资产到 `datasets/objdata_*`
2. 基于已准备好的资产构建 object-scale 索引
3. 采样表面点和抓取帧
4. 用 MuJoCo 做碰撞与稳定性筛选
5. 保存抓取状态到 `datasets/graspdata_*`
6. 用 Warp 渲染多视角部分点云
7. 按需构建形状聚类与 RL 专用元数据到 `_meta/`

当前同时支持：
- CPU MuJoCo 采样：`run.py` / `run_multi.py`
- GPU MJWarp 采样：`run_mjw.py` / `run_multi_mjw.py`

## 目录
- [仓库介绍](#仓库介绍)
- [安装与依赖](#安装与依赖)
- [数据集构造](#数据集构造)
- [聚类与 RL 元数据](#聚类与-rl-元数据)
- [运行指令](#运行指令)
- [工具脚本](#工具脚本)
- [主要文件详解](#主要文件详解)
- [注意事项](#注意事项)
- [Tuning Notes](#tuning-notes)
- [TODO](#todo)
- [License](#license)

## 仓库介绍

主线目标：
- 先准备可复用的物体资产
- 为 object-scale 条目离线生成抓取状态
- 用 MuJoCo 物理过滤不合理抓取
- 用统一格式保存抓取结果
- 在抓取完成后额外渲染部分点云，供下游学习使用
- 按需构建不依赖 grasp 文件的形状聚类与 RL 清单

当前主线约定：
- config-first
- manifest 先过滤，再进入索引
- `objdata_*` 和 `graspdata_*` 显式分离
- 全局 id 空间是 object-scale 粒度
- `grasp.h5` 是主结果，`grasp.npy` 由它导出
- 主线 grasp 数组按 `float32` 存储
- 数据集切分按 `object_name` 分组，避免不同 scale 间泄漏

默认约定：
- 默认配置：`configs/run_YCB_liberhand_right.json`
- 默认数据根目录：`assets/objects/processed`
- 生成数据集 tag 规则：`run_<...>.json -> graspdata_<...>`

## 安装与依赖

推荐环境：
- Ubuntu Linux
- Python 3.10
- MuJoCo
- Open3D / trimesh / h5py / tqdm / transforms3d
- 如需 Warp / MJWarp，则额外安装相应 GPU 依赖

推荐安装方式：

```bash
conda create -n grasp python=3.10 -y
conda activate grasp
pip install -r requirements.txt
```

如果要使用 Warp / MJWarp：
- 需要可用的 NVIDIA 驱动与 CUDA 运行环境
- 需要安装 `warp-lang`
- 需要安装 `mujoco_warp`

安装后的快速自检：

```bash
python run.py --help
python run_multi.py --help
python run_warp_render.py --help
python run_mjw.py --help
```

## 数据集构造

### 输入目录

`DatasetObjects` 会从这里索引处理后的物体数据：

```text
assets/objects/processed/
  YCB/
  DGN/
```

本仓库通常通过软链接接外部 mesh 仓库：

```text
assets/objects -> /home/ccs/repositories/mesh_process/assets/objects
```

### Manifest 过滤

只有 `manifest.process_meshes.json` 中满足：
- `process_status = success`

的物体才会进入索引。

### 物体资产准备

`prepare_object_assets.py` 是采样前必须先运行的准备入口。

它会：
- 从 `assets/objects/processed/<dataset>/manifest.process_meshes.json` 扫描可用物体
- 在 `datasets/objdata_<dataset>/` 下构建物体资产
- 写出 `coacd.obj`、`object.xml`、`convex_parts/*.obj`
- 写出 `pc_warp/global_pc.npy` 和 `pc_warp/global_normals.npy`
- 在配置启用时额外生成与 `scaleXXX` 平级的 `native/` 资产

示例：

```bash
python prepare_object_assets.py -c configs/assets_YCB.json --force --jobs 8
```

`DatasetObjects` 现在是针对已准备资产的只读索引器，不负责隐式重建资产。

### Object-Scale 扁平化

每个 `(object, scale)` 会变成一个独立条目，包含：
- `global_id`
- `object_scale_key`
- `object_name`
- `scale`
- `coacd_abs`
- `mjcf_abs`
- `output_dir_abs`

目前主线 liberhand 配置常用尺度列表：

```json
[0.08, 0.10, 0.12, 0.14, 0.16]
```

### 输出目录结构

当前主线将“物体资产”和“抓取输出”分开存放：

```text
datasets/objdata_YCB/<object>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/*.obj
  pc_warp/
    global_pc.npy
    global_normals.npy

datasets/graspdata_YCB_liberhand_right/<object>/scaleXXX/
  grasp.h5
  grasp.npy
  grasp_fail.h5
  grasp_fail.npy
  pc_warp/
    cam_in.npy
    cam_ex_XX.npy
    partial_pc_XX.npy
    partial_pc_cam_XX.npy
```

当前主线里，`grasp.h5` / `grasp.npy` 主要包含这些抓取数组：
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`
- `qpos_squeeze`

这些数组目前统一按 `float32` 存储。

当前回放语义补充：
- 保存的 `qpos_prepared` 仍然是原始候选 pregrasp 状态
- extforce 回放会使用 `qpos_squeeze` 的位姿加上保存的 prepared joints 重建验证用 pregrasp
- `sim_dataset.py` 保留保存的 `qpos_init` 预检查，然后调用 `sim_under_extforce(qpos_target, rebuilt_qpos_prepared, ...)`

失败样本补充：
- `run.py` 还会额外导出 `grasp_fail.h5` 和 `grasp_fail.npy`
- 其中保存 `qpos_fail` 和 `failure_stage`
- 当前保留的失败阶段包括 `prepared_contact`、`insufficient_contact` 和 `extforce_failure`
- `approach_contact` 和 `init_contact` 不进入失败数据集
- `qpos_fail` 保存该阶段对应的失败状态：
  - `prepared_contact`：发生碰撞的 prepared 状态
  - `insufficient_contact`：闭合后的 `qpos_grasp`
  - `extforce_failure`：失败的 `qpos_squeeze`
- 如果 `valid_count < data.min_valid_count`，则正样本和失败样本文件都会被截断为 0 行
- 否则失败样本会使用配置中的 `seed` 做 deterministic shuffle，并按 `floor(data.fail_keep_ratio * valid_count)` 截断保留

全局点云补充：
- `run.py` 会从 objdata 资产里读取 `global_pc.npy` / `global_normals.npy`
- `run_mjw.py` 会在抓取输出目录下写入 `pc_warp/global_pc.npy`
- 它保存的是抓取生成最开始采样到的世界系物体点云，不是 `partial_pc_XX.npy` 的拼接结果
- 当前主线默认 shape 为 `(sampling.n_points, 3)`，dtype 为 `float32`

### 数据集切分规则

`build_dataset_splits.py` 会在这里输出数据集清单：
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

当前切分规则：
- 按 `object_name` 切分，而不是按 object-scale 条目切分
- 同一个物体的所有 scale 必须留在同一个 split 中
- 默认按唯一物体数做约 `80/20` 切分，并使用配置中的 `seed` 打乱
- 只有正样本 grasp 输出、失败样本 grasp 输出和所需渲染输出都存在，条目才会进入清单
- 每条 manifest 会记录：
  - `grasp_h5_path`、`grasp_npy_path`
  - `grasp_h5_fail_path`、`grasp_fail_npy_path`
  - `global_pc_path`
- 写出最终 manifest 前会过滤空 grasp 文件
- manifest 中保存相对 dataset root 的相对路径，便于整体迁移

## 聚类与 RL 元数据

形状聚类和 RL 划分独立于模仿学习数据集构建流程。

### 形状聚类

使用 `run_shape_cluster.py` 基于已准备好的 objdata 资产构建 object-level 形状特征和 KMeans 聚类：

```bash
python run_shape_cluster.py -c configs/assets_YCB.json --force
```

当前约定：
- 使用预先保存的 `global_pc.npy`
- 使用一个配置的 scale tag，通常是 `scale120`
- 聚类是 object-level
- 不包含 `native`
- 所有输出只写到 `_meta`

主要输出目录：

```text
datasets/objdata_YCB/_meta/shape_cluster/<cluster_tag>/
```

主要文件：
- `meta.json`
- `object_features.npy`
- `cluster_centers.npy`
- `object_cluster.json`
- `cluster_index.json`
- `curriculum_index.json`

### RL 专用划分

使用 `build_dataset_splits_rl.py` 基于 objdata 资产和 shape-cluster 元数据构建 RL 专用 train/test 清单：

```bash
python build_dataset_splits_rl.py -c configs/assets_YCB.json --force
```

主要输出目录：

```text
datasets/objdata_YCB/_meta/rl_split/<split_tag>/
```

当前约定：
- 按 `object_name` 划分
- 再展开回 object-scale 条目
- 将 object-level 聚类信息继承到每条 object-scale 记录
- 当前只使用标准 `scaleXXX` 资产
- 不依赖 `grasp.h5` 或 Warp partial render 输出

### 聚类可视化

使用 `vis_shape_cluster.py` 生成按簇组织的缩略图页：

```bash
python vis_shape_cluster.py -c configs/assets_YCB.json
```

输出目录：

```text
datasets/objdata_YCB/_meta/shape_cluster/<cluster_tag>/vis/
```

## 运行指令

### CPU：单个 Object-Scale

`run.py` 负责单条目 CPU 采样。

```bash
python run.py \
  -c configs/run_YCB_liberhand_right.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand_right/YCB_002_master_chef_can/scale120 \
  --force -v
```

必填参数：
- `--object-scale-key`
- `--coacd-path`
- `--mjcf-path`
- `--output-dir`

### 可视化辅助脚本

在线抓取采样可视化，不做 extforce：

```bash
python demo_grasp.py \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  -c configs/run_YCB_liberhand_right.json -v
```

在线可视化，找到第一个通过 extforce 的抓取后停止：

```bash
python demo.py \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  -c configs/run_YCB_liberhand_right.json -v
```

对单个 object-scale 的所有已保存抓取做 MuJoCo extforce 回放可视化：

```bash
python vis_grasp_mujoco.py \
  --object-scale-dir datasets/graspdata_YCB_liberhand_right/YCB_002_master_chef_can/scale120 \
  -c configs/run_YCB_liberhand_right.json -v
```

### CPU：整数据集并行

`run_multi.py` 现在只负责并行跑抓取采样。

```bash
python run_multi.py -c configs/run_YCB_liberhand_right.json -j 16 --force
```

常用参数：
- `-j/--max-parallel`
- `--script`：替换为其他单条目脚本

## 工具脚本

### 检查索引结果

使用 `print_dataset_objects.py` 查看 `DatasetObjects` 条目，而不启动采样：

```bash
python print_dataset_objects.py -c configs/run_YCB_liberhand_right.json
python print_dataset_objects.py -c configs/run_YCB_liberhand_right_native.json --native-only
```
- `--force`
- `-v`：只透传给子进程 `run.py`

日志目录：

```text
logs/run/<dataset_tag>/
```

### GPU MJWarp：单个 Object-Scale

```bash
python run_mjw.py \
  -c configs/run_YCB_liberhand_right.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand_right/YCB_002_master_chef_can/scale120 \
  --batch-size 512 \
  --nconmax 32 \
  --naconmax 16384 \
  --njmax 200 \
  --ccd-iterations 200 \
  --force -v
```

### GPU MJWarp：整数据集并行

```bash
python run_multi_mjw.py \
  -c configs/run_YCB_liberhand_right.json \
  -j 4 \
  --batch-size 512 \
  --njmax 200 \
  --ccd-iterations 200 \
  --force
```

日志目录：

```text
logs/run_mjw/<dataset_tag>/
```

### Warp 部分点云渲染

`run_warp_render.py` 同时支持：
- 单条目模式：传 `-i` 或 `-k`
- 整数据集模式：两者都不传

整数据集：

```bash
python run_warp_render.py -c configs/run_YCB_liberhand_right.json -j 2
```

单条目：

```bash
python run_warp_render.py -c configs/run_YCB_liberhand_right.json -i 0
python run_warp_render.py -c configs/run_YCB_liberhand_right.json -k YCB_002_master_chef_can__scale080
```

### 构建数据集 Split

`build_dataset_splits.py` 会扫描已有抓取结果和渲染结果，写出：
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

```bash
python build_dataset_splits.py -c configs/run_YCB_liberhand_right.json
```

### 接触参数对比实验

`scripts/solimp_solref_experiment.py` 会在单个 object-scale 上跑四组 `solimp/solref` 对比：
- `hand_soft_obj_hard`
- `hand_hard_obj_soft`
- `hand_hard_obj_hard`
- `hand_soft_obj_soft`

默认输出：
- 生成的测试配置：`scripts/configs/solimp_solref_cases/`
- 实验日志与汇总：`tmp/solimp_solref_experiment/`

```bash
python scripts/solimp_solref_experiment.py \
  --object-scale-key YCB_013_apple__scale080 \
  --asset-dir datasets/objdata_YCB/YCB_013_apple/scale080 \
  --work-dir tmp/solimp_solref_experiment_YCB_013_apple_scale080 \
  --parallel-cases --max-workers 4
```

### 数据集仿真复验

`sim_dataset.py` 会基于 `train.json` / `test.json` 中的已保存 grasp，重新调用
`MjHO.sim_under_extforce` 做稳定性复验。

当前主线复验逻辑：
- 使用保存的 `qpos_squeeze`
- 在 extforce 前重做 `qpos_init` / `qpos_prepared` 的碰撞门控检查
- 默认按 `float32` 读取并转换 qpos，与主线数据格式一致

默认按 `float32` 读取并转换 qpos：

```bash
python sim_dataset.py -c configs/run_YCB_liberhand_right.json --split train -v
```

如果要比较 `float32` 和 `float64` 两种精度下的成功率：

```bash
python sim_dataset.py -c configs/run_YCB_liberhand_right.json --split train --dtype float32 -v
python sim_dataset.py -c configs/run_YCB_liberhand_right.json --split train --dtype float64 -v
```

### 一键流水线脚本

`scripts/run_pipeline.sh` 会依次执行：
1. 清页缓存
2. `run_multi.py`
3. `run_warp_render.py`
4. `build_dataset_splits.py`

```bash
bash scripts/run_pipeline.sh -c configs/run_YCB_liberhand_right.json
```

可选覆盖：

```bash
bash scripts/run_pipeline.sh \
  -c configs/run_YCB_liberhand_right.json \
  --cpu-set 0-23 \
  --run-j 24 \
  --render-j 2 \
  --no-force \
  --no-drop-caches
```

### 可视化

物体网格：

```bash
python vis_obj.py -c configs/run_YCB_liberhand_right.json -i 0
python vis_obj.py -c configs/run_YCB_liberhand_right.json -k YCB_002_master_chef_can__scale080
```

手物体姿态：

```bash
python vis_ho.py -c configs/run_YCB_liberhand_right.json -i 0
```

已保存抓取状态：

```bash
python vis_grasp.py -c configs/run_YCB_liberhand_right.json -i 0
python vis_grasp.py -c configs/run_YCB_liberhand_right.json -i 0 --vis-ids 0,10,-1 --frame-stage qpos_grasp
```

已保存部分点云：

```bash
python vis_pc.py -c configs/run_YCB_liberhand_right.json -i 0 --show-cam-frames
```

抓取姿态分布绘图：

```bash
PYTHONPATH=. python tools/visualization/plot_grasp_pose_plotly.py -c configs/run_YCB_liberhand_right.json -i 0
```

## 主要文件详解

### 入口脚本
- [run.py](/home/ccs/repositories/dexgrasp_sample/run.py)
  CPU 单条目抓取采样
- [run_multi.py](/home/ccs/repositories/dexgrasp_sample/run_multi.py)
  CPU 数据集级并行调度
- [run_mjw.py](/home/ccs/repositories/dexgrasp_sample/run_mjw.py)
  GPU MJWarp 单条目采样
- [run_multi_mjw.py](/home/ccs/repositories/dexgrasp_sample/run_multi_mjw.py)
  GPU MJWarp 数据集级并行调度
- [run_warp_render.py](/home/ccs/repositories/dexgrasp_sample/run_warp_render.py)
  Warp 部分点云渲染
- [build_dataset_splits.py](/home/ccs/repositories/dexgrasp_sample/build_dataset_splits.py)
  构建 `train.json` / `test.json`
- [sim_dataset.py](/home/ccs/repositories/dexgrasp_sample/sim_dataset.py)
  重放已保存数据集 grasp，并统计 `float32` / `float64` 下的 extforce 成功率
- [scripts/run_pipeline.sh](/home/ccs/repositories/dexgrasp_sample/scripts/run_pipeline.sh)
  一键完整流水线
- [scripts/solimp_solref_experiment.py](/home/ccs/repositories/dexgrasp_sample/scripts/solimp_solref_experiment.py)
  单物体接触参数对比实验（支持并行运行四组 case）

### 核心模块
- [src/dataset_objects.py](/home/ccs/repositories/dexgrasp_sample/src/dataset_objects.py)
  基于 manifest 的 object-scale 索引与资产定位
- [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py)
  CPU MuJoCo 手物体仿真辅助类
- [src/mjw_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mjw_ho.py)
  MJWarp 批量仿真辅助类
- [src/sample.py](/home/ccs/repositories/dexgrasp_sample/src/sample.py)
  抓取帧采样逻辑
- [utils/utils_sample.py](/home/ccs/repositories/dexgrasp_sample/utils/utils_sample.py)
  共享采样与输出辅助函数
- [utils/utils_file.py](/home/ccs/repositories/dexgrasp_sample/utils/utils_file.py)
  config / 路径 / 日志目录工具函数

### 可视化脚本
- [vis_obj.py](/home/ccs/repositories/dexgrasp_sample/vis_obj.py)
- [vis_obj_scale.py](/home/ccs/repositories/dexgrasp_sample/vis_obj_scale.py)
- [vis_ho.py](/home/ccs/repositories/dexgrasp_sample/vis_ho.py)
- [vis_grasp.py](/home/ccs/repositories/dexgrasp_sample/vis_grasp.py)
- [vis_pc.py](/home/ccs/repositories/dexgrasp_sample/vis_pc.py)

## 注意事项

- 主线是 config-first，缺字段应当直接报错，而不是依赖隐藏默认值。
- `run_multi.py` 现在只负责并行抓取采样，split 导出已经拆到 `build_dataset_splits.py`。
- `run_warp_render.py` 同时支持单条目模式和整数据集模式。
- `grasp.h5` 是主结果文件，`grasp.npy` 始终由它导出。
- 当前主线 `grasp.h5` / `grasp.npy` 中的 grasp 数组按 `float32` 保存。
- 当前主线数据格式包含 `qpos_init`、`qpos_approach`、`qpos_prepared`、`qpos_grasp`、`qpos_squeeze`。
- CPU extforce 验证采用两阶段流程：先检查无外力 settle 漂移，再以 settle 后位姿为基准施加六个方向的外力。
- `sim_dataset.py` 会复验保存的 `qpos_squeeze`，并支持把已保存 qpos 强制转成 `float32` 或 `float64` 后比较精度敏感性。
- `build_dataset_splits.py` 按 object 切分，而不是按 object-scale 行切分，以避免 scale 间泄漏。
- 当前工作流里 `docs/` 更多用于本地参考，通常不会上传到 git。
- CPU 和 GPU 采样使用统一的 object-scale 接口，但运行时行为和性能特征不同。
- MJWarp GPU 路径不保证严格确定性。

## Tuning Notes

### CPU
- `data.flush_every = 25` 是当前默认值，可以避免每接受一个有效样本就刷一次 HDF5。
- `run.py` 使用 `data.max_time_sec` 作为整个 CPU 采样流程的总 wall-clock 上限。
- `sim_grasp(record_history=False)` 是主路径默认值；只有可视化/调试才建议开启 history。
- 如果要做更干净的 CPU benchmark：

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0-15 python run_multi.py -c configs/run_YCB_liberhand_right.json -j 16
```

### MJWarp
- 当前 MJWarp 路线仍以 `data.max_cap = 100` 作为较实用的采样目标。
- `run_mjw.py` 使用 `data.max_time_sec = 180.0` 作为 extforce 阶段 wall-clock 上限。
- 如果目标是尽快拿到约 `100` 个有效抓取，`batch_size` 建议在 `128`、`256`、`512` 这类量级。
- `4096` 这种大 batch 更适合大规模抓取收集，不适合围绕 `100` 个有效抓取做快速早停。
- 当前主线默认 `sampling.n_points = 2048`，是 `max_cap = 100` 工作流下的折中值。

## TODO

当前后续方向：
- 提高 GPU 版本吞吐量，后续可能会探索“多个物体共用一个 batch”的方案，而不只是单物体单 batch。
- 考虑不同抓取构型，而不只局限于当前设置，例如 2 指、3 指、4 指抓取。

当前活动计划文件：
- [TODO.md](/home/ccs/repositories/dexgrasp_sample/TODO.md)

## References

- BODex: <https://github.com/JYChen18/BODex>
- DexGraspBench: <https://github.com/JYChen18/DexGraspBench>

## License

本项目使用 MIT License。
详见 [LICENSE](/home/ccs/repositories/dexgrasp_sample/LICENSE)。
