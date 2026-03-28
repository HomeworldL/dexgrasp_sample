# dexgrasp_sample

一个基于 MuJoCo 的离线灵巧手抓取采样仓库，面向“缩放后的 3D 物体 -> 抓取状态 -> 部分点云”的整套数据构造流程。

主线流水线是：

1. 基于 manifest 构建 object-scale 索引
2. 采样表面点和抓取帧
3. 用 MuJoCo 做碰撞与稳定性筛选
4. 保存抓取状态到 `grasp.h5` 与 `grasp.npy`
5. 用 Warp 渲染多视角部分点云
6. 构建按 object 切分的数据集级别 `train.json` / `test.json`

当前同时支持：
- CPU MuJoCo 采样：`run.py` / `run_multi.py`
- GPU MJWarp 采样：`run_mjw.py` / `run_multi_mjw.py`

## 目录
- [仓库介绍](#仓库介绍)
- [安装与依赖](#安装与依赖)
- [数据集构造](#数据集构造)
- [运行指令](#运行指令)
- [主要文件详解](#主要文件详解)
- [注意事项](#注意事项)
- [Tuning Notes](#tuning-notes)
- [TODO](#todo)
- [License](#license)

## 仓库介绍

主线目标：
- 为 object-scale 条目离线生成抓取状态
- 用 MuJoCo 物理过滤不合理抓取
- 用统一格式保存抓取结果
- 在抓取完成后额外渲染部分点云，供下游学习使用

当前主线约定：
- config-first
- manifest 先过滤，再进入索引
- 全局 id 空间是 object-scale 粒度
- `grasp.h5` 是主结果，`grasp.npy` 由它导出
- 主线 grasp 数组按 `float32` 存储
- 数据集切分按 `object_name` 分组，避免不同 scale 间泄漏

默认约定：
- 默认配置：`configs/run_YCB_liberhand.json`
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
  DGN2/
  HOPE/
```

本仓库通常通过软链接接外部 mesh 仓库：

```text
assets/objects -> /home/ccs/repositories/mesh_process/assets/objects
```

### Manifest 过滤

只有 `manifest.process_meshes.json` 中满足：
- `process_status = success`

的物体才会进入索引。

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

例如 `graspdata_YCB_liberhand` 下的输出大致是：

```text
datasets/graspdata_YCB_liberhand/<object>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/*.obj
  grasp.h5
  grasp.npy
  partial_pc_warp/
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

### 数据集切分规则

`build_dataset_splits.py` 会在这里输出数据集清单：
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

当前切分规则：
- 按 `object_name` 切分，而不是按 object-scale 条目切分
- 同一个物体的所有 scale 必须留在同一个 split 中
- 默认按唯一物体数做约 `80/20` 切分，并使用配置中的 `seed` 打乱
- 只有 `grasp.h5`、`grasp.npy` 和所需渲染输出都存在，条目才会进入清单
- 写出最终 manifest 前会过滤空 grasp 文件
- manifest 中保存相对 dataset root 的相对路径，便于整体迁移

## 运行指令

### CPU：单个 Object-Scale

`run.py` 负责单条目 CPU 采样。

```bash
python run.py \
  -c configs/run_YCB_liberhand.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120 \
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
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/object.xml \
  -c configs/run_YCB_liberhand.json -v
```

在线可视化，找到第一个通过 extforce 的抓取后停止：

```bash
python demo.py \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/object.xml \
  -c configs/run_YCB_liberhand.json -v
```

对单个 object-scale 的所有已保存抓取做 MuJoCo extforce 回放可视化：

```bash
python vis_grasp_mujoco.py \
  --object-scale-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120 \
  -c configs/run_YCB_liberhand.json -v
```

### CPU：整数据集并行

`run_multi.py` 现在只负责并行跑抓取采样。

```bash
python run_multi.py -c configs/run_YCB_liberhand.json -j 16 --force
```

常用参数：
- `-j/--max-parallel`
- `--script`：替换为其他单条目脚本
- `--force`
- `-v`：只透传给子进程 `run.py`

日志目录：

```text
logs/run/<dataset_tag>/
```

### GPU MJWarp：单个 Object-Scale

```bash
python run_mjw.py \
  -c configs/run_YCB_liberhand.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120 \
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
  -c configs/run_YCB_liberhand.json \
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
python run_warp_render.py -c configs/run_YCB_liberhand.json -j 2
```

单条目：

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -i 0
python run_warp_render.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

### 构建数据集 Split

`build_dataset_splits.py` 会扫描已有抓取结果和渲染结果，写出：
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

```bash
python build_dataset_splits.py -c configs/run_YCB_liberhand.json
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
python sim_dataset.py -c configs/run_YCB_liberhand.json --split train -v
```

如果要比较 `float32` 和 `float64` 两种精度下的成功率：

```bash
python sim_dataset.py -c configs/run_YCB_liberhand.json --split train --dtype float32 -v
python sim_dataset.py -c configs/run_YCB_liberhand.json --split train --dtype float64 -v
```

### 一键流水线脚本

`scripts/run_pipeline.sh` 会依次执行：
1. 清页缓存
2. `run_multi.py`
3. `run_warp_render.py`
4. `build_dataset_splits.py`

```bash
bash scripts/run_pipeline.sh -c configs/run_YCB_liberhand.json
```

可选覆盖：

```bash
bash scripts/run_pipeline.sh \
  -c configs/run_YCB_liberhand.json \
  --cpu-set 0-23 \
  --run-j 24 \
  --render-j 2 \
  --no-force \
  --no-drop-caches
```

### 可视化

物体网格：

```bash
python vis_obj.py -c configs/run_YCB_liberhand.json -i 0
python vis_obj.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

手物体姿态：

```bash
python vis_ho.py -c configs/run_YCB_liberhand.json -i 0
```

已保存抓取状态：

```bash
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0 --vis-ids 0,10,-1 --frame-stage qpos_grasp
```

已保存部分点云：

```bash
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0 --show-cam-frames
```

抓取姿态分布绘图：

```bash
PYTHONPATH=. python tools/visualization/plot_grasp_pose_plotly.py -c configs/run_YCB_liberhand.json -i 0
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
- [vis_partial_pc.py](/home/ccs/repositories/dexgrasp_sample/vis_partial_pc.py)

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
- `output.flush_every = 25` 是当前默认值，可以避免每接受一个有效样本就刷一次 HDF5。
- `run.py` 使用 `output.max_time_sec` 作为整个 CPU 采样流程的总 wall-clock 上限。
- `sim_grasp(record_history=False)` 是主路径默认值；只有可视化/调试才建议开启 history。
- 如果要做更干净的 CPU benchmark：

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0-15 python run_multi.py -c configs/run_YCB_liberhand.json -j 16
```

### MJWarp
- 当前 MJWarp 路线仍以 `output.max_cap = 100` 作为较实用的采样目标。
- `run_mjw.py` 使用 `output.max_time_sec = 180.0` 作为 extforce 阶段 wall-clock 上限。
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
