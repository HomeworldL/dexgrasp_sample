# dexgrasp_sample

基于 MuJoCo 的多数据集物体离线灵巧手抓取构型采样与点云渲染仓库。

## 1. 项目介绍
本仓库用于从物体网格生成可用抓取构型（离线数据生成流程）。

主流程：
1. 从处理后的数据集中构建 object-scale 资产。
2. 采样物体表面点与法向。
3. 采样抓取候选坐标系并转换为手基座位姿。
4. 使用 MuJoCo 执行分阶段碰撞过滤、闭合抓取与外力稳定性验证。
5. 将有效抓取写入 `grasp.h5`。
6. 从 `grasp.h5` 导出 `grasp.npy`。
7. （可选后处理）使用 Warp 渲染每个 object-scale 的多视角部分点云。

主线代码（来自 `AGENTS.md`）：
- `run.py`
- `run_multi.py`
- `run_warp_render.py`
- `src/mj_ho.py`
- `src/sample.py`
- `src/dataset_objects.py`

---

## 2. 依赖
安装基础依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：
- `numpy`, `scipy`, `torch`
- `trimesh`, `open3d`
- `mujoco`, `mujoco-viewer`
- `h5py`, `tqdm`, `transforms3d`
- `viser`, `pytest`

按功能补充可选依赖：
- Warp 部分点云渲染：
  - `warp-lang`
  - 具备 headless EGL/OpenGL 的运行环境
- Plotly 抓取坐标系可视化：
  - `plotly`

---

## 3. 主要文件与功能
### 3.1 流水线入口脚本
- `run.py`
  - 单个 object-scale 的抓取采样 worker。
  - 必须显式传入 object-scale 信息（`--object-scale-key`、`--coacd-path`、`--mjcf-path`、`--output-dir`）。
- `run_multi.py`
  - 并行调度器，从配置数据集遍历所有 object-scale，并调用 `run.py`。
- `run_warp_render.py`
  - 抓取后处理：为 object-scale 渲染并保存 Warp 部分点云。
  - 并行粒度是 object-scale（不是把同一物体的多个视角拆到不同 worker）。

### 3.2 核心模块
- `src/dataset_objects.py`
  - 基于 manifest 的多数据集 object-scale 索引。
  - 读取 `manifest.process_meshes.json` 并仅保留 `process_status=success`。
  - 在 `datasets/<dataset_tag>/<object>/scaleXXX/` 预构建缩放资产。
  - `dataset_tag` 由配置名 stem 派生：将前缀 `run_` 替换为 `graspdata_`。
- `src/scale_dataset_builder.py`
  - 生成每个 scale 的：
    - `coacd.obj`
    - `convex_parts/*.obj`
    - `object.xml`
- `src/sample.py`
  - 抓取帧采样、点云 FPS 等工具。
- `src/mj_ho.py`
  - MuJoCo 手-物仿真核心：
    - 接触检测
    - 抓取闭合仿真
    - 外力稳定性验证

### 3.3 根目录可视化入口
- `vis_obj.py`
  - 在 Viser 显示物体 mesh + 点云，然后可进入 MuJoCo 暂停视图。
- `vis_ho.py`
  - 在 Viser 显示手-物姿态，然后可进入 MuJoCo 暂停视图。
- `vis_grasp.py`
  - 在 Viser 显示物体 + 指定抓取样本（四阶段 qpos）的手 mesh。
  - 同时可用 Plotly 显示所有抓取坐标系。
- `vis_partial_pc.py`
  - 显示 Warp 保存的 `partial_pc_XX.npy`（世界系）与 `partial_pc_cam_XX.npy`（相机系）。
  - 可选显示 `cam_ex_XX.npy` 对应相机坐标系。

### 3.4 `tools/visualization` 脚本说明
- 目录：`tools/visualization/`
- 主要用于研究/调试。
- `plot_grasp_pose_plotly.py` 已支持配置驱动。
- 其他不少脚本仍是早期硬编码实验脚本，通常需要先改路径再用。

---

## 4. 数据集接口与输出目录
### 4.1 输入数据根目录
由配置 `dataset.root` 指定，推荐：
- `assets/objects/processed`

常见软链接方式：
- `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`

### 4.2 manifest 过滤规则
仅纳入：
- `process_status=success`

### 4.3 object-scale 扁平索引
`DatasetObjects` 输出全局索引，每条记录包含：
- `global_id`
- `object_name`
- `coacd_abs`
- `convex_parts_abs`
- `scale`
- `mjcf_abs`
- `output_dir_abs`
- `object_scale_key`

### 4.4 产物目录结构
每个 object-scale 对应：

```text
datasets/<dataset_tag>/<object_name>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/part_000.obj ...
  grasp.h5
  grasp.npy
  <warp_render.output_subdir>/
    cam_in.npy
    cam_ex_00.npy ...
    partial_pc_00.npy ...
    partial_pc_cam_00.npy ...
```

说明：
- `run.py` 固定写 `grasp.h5` 并导出 `grasp.npy`。
- 点云数据与抓取数组分开存储。

---

## 5. 抓取数据格式（`grasp.h5`）
H5 元信息数据集：
- `object_name`（字符串）
- `scale`（float32）
- `hand_name`（字符串）
- `rot_repr`（`wxyz+qpos`）

核心 qpos 数据集：
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`

每行格式：
- `[tx, ty, tz, qw, qx, qy, qz, q1...qN]`

语义：
- `qpos_prepared`：基座位姿 + prepared 手指关节
- `qpos_approach`：基座位姿 + approach 手指关节
- `qpos_init`：approach 关节 + 位姿偏移后的初始接近状态
- `qpos_grasp`：闭合后抓取状态

---

## 6. 主要 CLI 使用方法
### 6.1 `run.py`
单个 object-scale 抓取采样：

```bash
python run.py \
  -c configs/run_YCB_liberhand.json \
  --object-scale-key YCB_002_master_chef_can__scale080 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale080/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale080/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale080 \
  -v
```

参数：
- 必填：`--object-scale-key`、`--coacd-path`、`--mjcf-path`、`--output-dir`
- `object_name` 和 `scale` 由 `--object-scale-key` 解析得到
- 可选：`-c/--config`（默认 `configs/run_YCB_liberhand.json`）
- 可选：`-v/--verbose`

### 6.2 `run_multi.py`
并行遍历配置中的所有 object-scale：

```bash
python run_multi.py -c configs/run_YCB_liberhand.json -j 4 -v
```

参数：
- `-j/--max-parallel`：最大并行子进程数
- `--script`：每条任务调用的脚本（默认 `run.py`）
- `-c/--config`
- `-v/--verbose`

`run_multi.py` 现在只负责并行抓取采样。

### 6.2.1 `build_dataset_splits.py`
基于已有抓取结果和点云渲染结果，构建数据集级别的 `train.json` / `test.json`。

```bash
python build_dataset_splits.py -c configs/run_YCB_liberhand.json
```

该脚本会写出：
- `train.json`
- `test.json`

构建规则：
- 先重新遍历当前 config 对应的全部 object-scale 条目。
- 仅收录产物完整的条目：需要存在 `grasp.h5`、`grasp.npy`、`cam_in.npy`，以及一一对应的 `partial_pc_XX.npy`、`partial_pc_cam_XX.npy`、`cam_ex_XX.npy`。
- 按 `object_name` 分组做稳定的 4:1 划分，同一个物体的全部 scale 只会落在同一个 split。

`train.json` / `test.json` 里的每条记录仍然是 obj-scale 粒度，路径都相对 `datasets/<dataset_tag>/`，字段包括：
- `global_id`
- `object_scale_key`
- `object_name`
- `output_path`
- `coacd_path`
- `mjcf_path`
- `grasp_h5_path`
- `grasp_npy_path`
- `partial_pc_path`
- `partial_pc_cam_path`
- `cam_ex_path`
- `cam_in`
- `scale`

### 6.3 `run_warp_render.py`
抓取后处理：渲染部分点云。

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -j 1 -v
```

单条目示例：

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -i 0
python run_warp_render.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

参数：
- `-c/--config`
- `-i/--obj-id` 或 `-k/--obj-key`
- `-j/--max-parallel`
- `--gpu-lst`：覆写设备列表（如 `0,1` 或 `cpu`）
- `--force`：忽略 skip-existing 强制重渲染
- `-v/--verbose`

并行语义：
- 按 object-scale 分给 worker。
- 每个条目的多视角在同一 worker 内完成。

### 6.4 `eval_dataset.py`
直接验证数据集 `train.json` 或 `test.json` 里已有 `qpos_grasp` 的仿真成功率，不经过网络。

```bash
python eval_dataset.py \
  -c configs/run_YCB_liberhand.json \
  --split test \
  --visualize \
  -v
```

说明：
- split 文件路径由 `config` 与 `--split` 自动解析到 `datasets/<dataset_tag>/<split>.json`。
- 每个 obj-scale 直接读取 `grasp.h5` 中的 `qpos_grasp`。
- 对每条 `qpos_grasp` 先检查 `qpos_init` 与 `qpos_prepared` 是否接触；任一接触即判失败。
- 若前置检查通过，再调用 `MjHO.sim_under_extforce`。
- `--visualize` 会打开 MuJoCo viewer，显示前置接触检查与外力稳定性验证过程。
- 输出每个 obj-scale 和整个 split 的成功数、总数、成功率。
- 不保存新的抓取结果文件。

---

## 7. 可视化脚本与使用方法
### 7.1 `vis_obj.py`
```bash
python vis_obj.py -c configs/run_YCB_liberhand.json -i 0
# 或
python vis_obj.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

### 7.2 `vis_ho.py`
```bash
python vis_ho.py -c configs/run_YCB_liberhand.json -i 0
```

### 7.3 `vis_grasp.py`
```bash
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0 --vis-ids 0,10,-1 --frame-stage qpos_grasp
```

常用参数：
- `--grasp-path`：显式指定 `.h5` / `.npy`
- `--skip-plotly`
- `--plotly-html out.html`

### 7.4 `vis_partial_pc.py`
```bash
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0 --view-ids 0,1 --show-cam-frames
```

常用参数：
- `--pc-subdir`：覆写点云子目录
- `--hide-mesh`
- `--show-cam-frames`

### 7.5 `tools/visualization/plot_grasp_pose_plotly.py`
建议在仓库根目录加模块路径运行：

```bash
PYTHONPATH=. python tools/visualization/plot_grasp_pose_plotly.py -c configs/run_YCB_liberhand.json -i 0
```

---

## 8. Config JSON 内容说明
配置文件命名：
- `configs/run_<dataset_group>_<hand>.json`

示例：
- `run_YCB_liberhand.json`
- `run_DGN2_liberhand2.json`
- `run_HOPE_inspire.json`

### 8.1 `dataset`
- `root`：数据根目录
- `include`：数据集列表
- `scales`：固定尺度列表
- `verbose`：索引阶段日志

### 8.2 `hand`
- `xml_path`
- `prepared_joints`
- `approach_joints`
- `shift_local`
- `transform`
  - `base_rot_grasp_to_palm`
  - `extra_euler`（`axis`, `degrees`）
- `target_body_params`（接触/距离权重）

### 8.3 `sampling`
- `n_points`：表面采样点数
- `downsample_for_sim`：用于仿真辅助点云数量
- `Nd`, `rot_n`, `d_min`, `d_max`, `max_points`：抓取帧采样控制项

### 8.4 `sim_grasp`
- 抓取闭合仿真参数（`Mp`, `steps`, `speed_gain`, `max_tip_speed`）
- `contact_min_count`：进入外力稳定性验证前的最小手物接触数

### 8.5 `extforce`
- 外力稳定性验证参数（时长、阈值、力大小、检查步长）
- `run_mjw.py` 将 `output.max_time_sec` 作为 extforce 阶段的 wall-clock 上限
- `run.py` 将 `output.max_time_sec` 作为整个 CPU 采样流程的总 wall-clock 上限
- 当前推荐默认值是 `output.max_time_sec = 180.0`

### 8.6 `output`
- `max_cap`, `max_time_sec`, `flush_every`, `h5_name`, `npy_name`
- `dataset_root` 为可选项，供 `run_multi.py` 等数据集级脚本使用
- 当前实现在每个条目目录写 `grasp.h5`。
- 当前实现固定从 `grasp.h5` 导出 `grasp.npy`。

### 8.7 `warp_render`
- 设备与并行：
  - `gpu_lst`, `thread_per_gpu`
- 输出与跳过：
  - `output_subdir`（当前默认 `partial_pc_warp`）
  - `save_pc/save_rgb/save_depth`
  - `skip_existing`
- 渲染几何参数：
  - `tile_width`, `tile_height`, `n_cols`, `n_rows`, `z_near`, `z_far`
- 相机模型参数：
  - `intrinsics`（`preset/fx/fy/cx/cy`）
  - `camera`（`type/radius/center/noise/lookat/up`）

---

## 9. 部分点云说明
- 当前 `cam_ex_XX.npy` 语义是 camera-to-world 变换。
- `partial_pc_XX.npy` 来自 depth 反投影（内参）再经外参变换到世界坐标。
- `partial_pc_cam_XX.npy` 保存同一批采样点在相机坐标系下的表示。
- 因此点云几何同时依赖内参与外参。

---

## 10. 快速运行命令
```bash
# 1) 抓取采样（全量 object-scale）
python run_multi.py -c configs/run_YCB_liberhand.json -j 4

# 2) 抓取后渲染部分点云
python run_warp_render.py -c configs/run_YCB_liberhand.json -j 1

# 3) 可视化某个 object-scale 的部分点云
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0 --show-cam-frames
```

给定一个 config 一键跑完整流水线：

```bash
bash scripts/run_pipeline.sh -c configs/run_YCB_liberhand.json
```

默认行为：
- 先清 Linux 页缓存
- 用 `taskset -c 0-15`、`-j 16`、`--force` 运行 `run_multi.py`
- 用 `-j 2` 运行 `run_warp_render.py`
- 最后运行 `build_dataset_splits.py`

可选覆盖参数：

```bash
bash scripts/run_pipeline.sh \
  -c configs/run_YCB_liberhand.json \
  --cpu-set 0-23 \
  --run-j 24 \
  --render-j 2 \
  --no-force \
  --no-drop-caches
```

---

## 11. CPU 基准运行建议
这几条主要用于在运行 `run_multi.py` 前做更干净的 CPU 基准测试。

清理 Linux 页缓存（需要 `sudo`）：

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
```

用 `taskset` 绑定 `run_multi.py` 及其所有子进程到指定 CPU 核：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0-15 python run_multi.py -c configs/run_YCB_liberhand.json -j 16
```

如果机器是 NUMA，优先用 `numactl`，同时约束 CPU 和本地内存分配：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
numactl --physcpubind=0-15 --localalloc \
python run_multi.py -c configs/run_YCB_liberhand.json -j 16
```

说明：
- `run_multi.py` 拉起的子进程会继承 `taskset` 的 CPU 亲和度。
- 把 BLAS/OpenMP 线程数限制为 `1`，可以避免 `run_multi.py` 已经做进程并行时再次线程过度竞争。
- 清页缓存主要用于提升基准测试可重复性，不是日常生产运行的必需步骤。

---

## 12. CPU 优化说明
- `run.py` 现在会从 config 读取 `output.flush_every`，只做周期性 `grasp.h5` 刷盘，而不是每接收一个有效样本就刷一次。
- 对当前 `max_cap = 100` 的流程，推荐默认值是 `output.flush_every = 25`。这通常只会产生少量中途 flush，再加最后一次必定执行的收尾 flush。
- 即使最后剩余样本数不足 `flush_every`，也仍然会写入，因为 `run.py` 在退出 HDF5 上下文前一定会做一次最终 resize + `hf.flush()`。
- 抓取闭合阶段现在默认使用 `sim_grasp(record_history=False)`，高频主路径不再保存逐步 history。只有可视化或调试脚本才应显式打开 `record_history=True`。
- `run.py` 的接触数过滤现在走更轻量的 `MjHO.get_contact_num()`，不再在主路径里构造完整的 `get_contact_info()` 字典结构。
- `MjHO` 初始化时还会预计算 `ctrl` 映射索引和侧摆自由度索引，因此 `qpos2ctrl()` 以及反复的侧摆掩码不再每一步都重建相同索引列表。
- `get_hand_qpos(copy=True)` 现在支持非复制视图模式，后续高频路径可以按需复用；当前主循环只在确实需要防止原地修改时才做显式拷贝。

---

## 13. 测试
```bash
pytest -q
```
