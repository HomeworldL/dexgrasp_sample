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
6. （可选后处理）使用 Warp 渲染每个 object-scale 的多视角部分点云。

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
  - 在 `datasets/<config_stem>/<object>/scaleXXX/` 预构建缩放资产。
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
  - 显示 Warp 保存的 `partial_pc_XX.npy`。
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
datasets/<config_stem>/<object_name>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/part_000.obj ...
  grasp.h5
  <warp_render.output_subdir>/
    cam_in.npy
    cam_ex_00.npy ...
    partial_pc_00.npy ...
```

说明：
- 当前 `run.py` 实际直接写 `grasp.h5`。
- 点云数据与抓取数组分开存储。

---

## 5. 抓取数据格式（`grasp.h5`）
核心数据集：
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
  --coacd-path datasets/run_YCB_liberhand/YCB_002_master_chef_can/scale080/coacd.obj \
  --mjcf-path datasets/run_YCB_liberhand/YCB_002_master_chef_can/scale080/object.xml \
  --output-dir datasets/run_YCB_liberhand/YCB_002_master_chef_can/scale080 \
  -v
```

参数：
- 必填：`--object-scale-key`、`--coacd-path`、`--mjcf-path`、`--output-dir`
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

### 8.2 `object`
- `id`：可视化脚本的默认全局对象 id

### 8.3 `sampling`
- `n_points`：表面采样点数
- `downsample_for_sim`：用于仿真辅助点云数量
- `Nd`, `rot_n`, `d_min`, `d_max`, `max_points`：抓取帧采样控制项

### 8.4 `transform`
- `base_rot_grasp_to_palm`
- `extra_euler`（`axis`, `degrees`）

### 8.5 `hand`
- `xml_path`
- `prepared_joints`
- `approach_joints`
- `shift_local`
- `target_body_params`（接触/距离权重）

### 8.6 `validation`
- `contact_min_count`

### 8.7 `sim_grasp`
- 抓取闭合仿真参数（`Mp`, `steps`, `speed_gain`, `max_tip_speed`）

### 8.8 `extforce`
- 外力稳定性验证参数（时长、阈值、力大小、检查步长）

### 8.9 `output`
- `base_dir`, `max_cap`, `h5_name`, `npy_name`
- 当前实现实际按条目目录写 `grasp.h5`。

### 8.10 `warp_render`
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

---

## 11. 测试
```bash
pytest -q
```
