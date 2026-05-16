# AGENTS_zh.md

## 仓库用途
基于 MuJoCo 的 3D 物体灵巧抓取采样仓库。
当前主线工作分为三部分：
- 物体资产准备（`objdata_*`）
- 离线抓取配置生成（`graspdata_*`）
- 可选的形状聚类与 RL 专用元数据构建（位于 `_meta/` 下）

## 主线代码（不要漂移）
- `prepare_object_assets.py`
- `prepare_object_usds.py`
- `run_multi.py`
- `run.py`
- `run_warp_render.py`
- `src/mj_ho.py`（核心，保持稳定）
- `src/sample.py`
- `src/dataset_objects.py`

## 架构说明
- 当前数据流分成两个阶段：
  - 准备阶段：原始 processed mesh -> `prepare_object_assets.py` -> `datasets/objdata_*`
  - 采样阶段：`objdata_*` 资产 -> 抓取帧采样 -> MuJoCo 碰撞/稳定性过滤 -> `datasets/graspdata_*`
- USD 导出是显式的准备后步骤：
  - `prepare_object_usds.py` 会把已准备好的 `object.xml` 原地转换成可供 IsaacLab / Isaac Sim 使用的 USD 资产，仍写回 `datasets/objdata_*`
- 采样后的视觉流程：每个 object-scale 完成抓取采样后，运行 `run_warp_render.py`，基于缩放后的 `coacd.obj` 渲染多视角局部点云。
- 资产准备由 `prepare_object_assets.py` 负责；它直接基于 manifest 允许的源 mesh 构建 `objdata_*` 资产，并写出 `global_pc.npy` / `global_normals.npy`。
- 物体资产和抓取输出是显式分离的：
  - 物体资产位于 `datasets/objdata_*`
  - 抓取输出位于 `datasets/graspdata_*`
- `print_dataset_objects.py` 是 `DatasetObjects` 的轻量检查入口；用它可以在不启动采样的情况下确认 object-scale 扁平索引以及是否包含 native。
- 形状聚类与 RL 划分工具独立于模仿学习数据集构建，并且只写入 `_meta`：
  - `run_shape_cluster.py` 在 `datasets/<objdata_tag>/_meta/shape_cluster/` 下构建 object-level shape embedding 与 KMeans 聚类
  - `build_dataset_splits_rl.py` 在 `datasets/<objdata_tag>/_meta/rl_split/` 下构建 RL 专用 train/test manifest
  - `vis_shape_cluster.py` 为每个 cluster 渲染缩略图页，便于快速人工检查

## 数据集准备
- `prepare_object_assets.py` 是采样前必须先运行的入口。
- 它会扫描 `manifest.process_meshes.json`，过滤 `process_status=success`，并在 `datasets/objdata_*` 下构建物体资产。
- 每个准备好的资产目录包含：
  - `coacd.obj`
  - `object.xml`
  - `convex_parts/*.obj`
  - `pc_warp/global_pc.npy`
  - `pc_warp/global_normals.npy`
- 当 RL / IsaacLab 需要可模拟 USD 时，后续还需要运行 `prepare_object_usds.py`。
- USD 转换后，每个资产目录还可能包含：
  - `object.usd`
  - `configuration/object_base.usd`
  - `configuration/object_physics.usd`
  - `configuration/object_robot.usd`
  - `configuration/object_sensor.usd`
  - `config.yaml`
  - `.asset_hash`
- 标准缩放资产使用 `scaleXXX/` 目录。
- 当开关启用时，`native/` 会作为 `scaleXXX/` 的平级目录创建。
- `DatasetObjects` 是针对已准备好资产的只读索引器；它不能隐式重建资产。

## 数据集接口
- 首选数据集根目录为 `assets/objects/processed`。
- 本仓库通过符号链接接入外部 mesh 仓库：
  - `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`
- 主线默认合并数据集列表：
  - `["YCB"]`
- `DatasetObjects` 在合并后的数据集列表之上暴露全局整数 id 空间。
- `run.py` 通过 `--object-scale-key` 采样单个条目；`run_multi.py` 按 `global_id` 顺序遍历全部条目。
- `print_dataset_objects.py` 应作为检查索引结果和 scale/native 包含情况的首选工具。

## 缩放策略
- 所有数据集统一使用配置中的固定 scale 列表（`data.run_scales`）。
- 当前主线中 liberhand 配置的默认列表为：
  - `[0.08, 0.10, 0.12, 0.14, 0.16]`
- scale 粒度为 object-scale 扁平索引（每个物体的每个 scale 各对应一项）。
- 必须先经过 manifest 过滤：只有 `manifest.process_meshes.json` 中 `process_status=success` 的条目才可进入流程。

## 采样流水线（CPU版本）
- 为 `numpy/random/torch/open3D` 设置确定性随机种子（包括 CUDA 与 cuDNN 的确定性标志）。
- 通过 `DatasetObjects` 加载物体元数据和 mesh，并采样物体点云与法向（poisson）。
- 从点云/法向中采样候选抓取坐标系（`sample_grasp_frames`），然后将坐标系位姿转换到手模型位姿约定：
  - 应用 `hand.transform` 中的 grasp-to-palm 旋转对齐
  - 将旋转转换为四元数，并以 `wxyz` 顺序存储
  - 构建基础手位姿 `[tx,ty,tz,qw,qx,qy,qz]`
- 为每个候选构建三个预抓取状态：
  - `qpos_prepared`：基础位姿 + prepared 手指关节
  - `qpos_approach`：相同基础位姿 + approach 手指关节
  - `qpos_init`：在 approach 关节基础上，施加局部负 z 偏移并转换到世界坐标
- 在 MuJoCo（`MjHO`）中对每个候选执行分阶段碰撞过滤：
  - 若 `prepared` 碰撞则拒绝
  - 若 `approach` 碰撞则拒绝
  - 若 `init` 碰撞则拒绝
- 对于无碰撞候选：
  - 模拟闭合抓取得到 `qpos_grasp`
  - 要求足够的手-物体接触数（`sim_grasp.contact_min_count`，当前默认 `>=4`）
  - 基于 `qpos_grasp` 叠加 `extforce.grip_delta` 构建 `qpos_squeeze`
  - 在第二个模拟器中执行两阶段外力稳定性验证（`object_fixed=False`）：
    - 先用 `qpos_squeeze` 的位姿和保存的 prepared joints 重建验证用 `qpos_prepared`
    - 从重建后的 `qpos_prepared` 向 `qpos_squeeze` 闭合，并进行无外力 settle 漂移门控
    - 再以 settle 后位姿为基准施加六个方向的外力
  - 仅保留验证通过的抓取
- 先以 HDF5 格式持久化输出（`grasp.h5`）：
  - 数据集字段：`qpos_init`、`qpos_approach`、`qpos_prepared`、`qpos_grasp`、`qpos_squeeze`
  - 主线中抓取数组存储 dtype 为 `float32`
  - 使用预分配容量并在运行时截断到最终有效长度
  - 长时间运行时周期性 flush/GC
- 失败样本单独持久化为 `grasp_fail.h5` 和 `grasp_fail.npy`：
  - 字段：`qpos_fail`、`failure_stage`
  - 当前保留的失败阶段为 `prepared_contact`、`insufficient_contact` 和 `extforce_failure`
  - `approach_contact` 和 `init_contact` 不进入失败数据集
  - `qpos_fail` 保存该阶段对应的失败状态：
    - `prepared_contact` 保存发生碰撞的 prepared 状态
    - `insufficient_contact` 保存闭合后的 `qpos_grasp`
    - `extforce_failure` 保存失败的 `qpos_squeeze`
  - 如果 `valid_count < data.min_valid_count`，则正样本和失败样本文件都必须截断为 0 行
  - 否则失败样本使用配置中的 seed 做 deterministic shuffle，并截断到 `floor(data.fail_keep_ratio * valid_count)`
- `grasp.h5` 完成后，加载同一批数组并转换为 `grasp.npy`（数值必须与 HDF5 完全一致）。
- `sim_dataset.py` 是基于 `train.json` / `test.json` 的数据集级回放/验证入口。
  - 保留保存的 `qpos_init` 预检查
  - 使用保存的 `qpos_squeeze` 位姿和 prepared joints 重建回放用 `qpos_prepared`
  - 通过 `sim_under_extforce(qpos_target, rebuilt_qpos_prepared, ...)` 对保存的 `qpos_squeeze` 做复验
  - 支持 `--dtype float32` 或 `--dtype float64`，用于比较对精度敏感的 extforce 成功率
- 抓取输出完成后，使用 Warp 渲染物体局部点云，并保存到：
  - `datasets/<dataset_tag>/<object>/scaleXXX/<warp_render.output_subdir>/`
  - `dataset_tag` 规则：将配置文件 stem 前缀 `run_` 替换为 `graspdata_`
  - 默认子目录：`pc_warp`
  - `run.py` 会从 objdata 资产中读取预构建的 `global_pc.npy` 与 `global_normals.npy`
  - `run_mjw.py` 会在采样过程中把 `global_pc.npy` 保存到同一 `pc_warp/` 目录下
  - `global_pc.npy` 保存的是采样器最初使用的世界系物体点云，不是多视角 partial point cloud 的拼接
  - 当前主线默认是世界系 `float32`、shape 为 `(sampling.n_points, 3)`
- 点云单独保存，不得打包进 `grasp.npy`。

## 采样流水线（GPU版本）
- 使用MJWarp
- YCB 的 GPU harvesting 路径使用 `configs/run_YCB_liberhand_right_gpu.json`。
- GPU harvesting 配置目标为 `data.max_cap=2000`；GPU 路径不应使用 `data.max_time_sec` 截断 extforce 验证。
- MJWarp 运行容量参数仍是 CLI 选项，当前默认值为：
  - `--batch-size 256`
  - `--nconmax 32`
  - `--naconmax 16384`
  - `--njmax 200`
  - `--ccd-iterations 200`
- 只有在确认 GPU 显存和单步耗时可接受后再增大 `batch_size`；更大的 batch 能提高占用率，但也会让每步 MJWarp 和 extforce 循环更重。
- `batch_size=4096` 更适合大规模 harvesting 实验，不适合快速 sanity check。
- 当前 GPU 配置使用 `sampling.n_points=4096` 来保证候选覆盖率。
- GPU 失败样本导出会使用配置 seed 做确定性 shuffle，并截断到 `floor(data.fail_keep_ratio * valid_count)`。
- GPU 正样本输出当前不会因为 `data.min_valid_count` 被清零；该最小数量截断逻辑仍仅用于 CPU，除非后续明确启用。

## 数据集切分策略
- 使用 `build_dataset_splits.py` 扫描已完成的抓取/渲染输出，并生成 `datasets/<dataset_tag>/train.json` 与 `datasets/<dataset_tag>/test.json`。
- 按 `object_name` 做切分，而不是按 object-scale 条目切分；同一物体的所有 scale 必须留在同一个 split 中，避免泄漏。
- 默认按唯一物体数做约 `80/20` 切分，并使用配置中的 `seed` 打乱。
- 只有当正样本 grasp 输出、失败样本 grasp 输出以及所需渲染输出都存在时，该条记录才可进入 split。
- split 记录必须包含：
  - `grasp_h5_path`、`grasp_npy_path`
  - `grasp_h5_fail_path`、`grasp_fail_npy_path`
  - `global_pc_path`
- 空的 grasp 文件在最终 manifest 使用前必须过滤掉。
- manifest 中的路径应保存为相对 dataset root 的相对路径，便于数据集整体迁移。

## 形状聚类与 RL 元数据
- `run_shape_cluster.py` 用于 RL 导向的物体聚类，不属于模仿学习数据集构建流程。
- 当前形状聚类：
  - 使用 `objdata_*` 中预先保存的 `global_pc.npy`
  - 按一个配置的 scale tag 提取 object-level 特征，通常是 `scale120`
  - 不包含 `native`
  - 全部输出写入 `datasets/<objdata_tag>/_meta/shape_cluster/<cluster_tag>/`
- 主要聚类产物包括：
  - `meta.json`
  - `train_history.json`
  - `ae_state_dict.pt`
  - `object_labels.json`
  - `cluster_labels.json`
- `build_dataset_splits_rl.py` 会在 `datasets/<objdata_tag>/_meta/rl_split/<split_tag>/` 下写出 RL 专用 train/test manifest。
- RL 划分先按 `object_name` 分组，再展开回 object-scale 记录，并继承 object-level 聚类信息。
- 当前 RL 划分仅针对标准缩放资产，不包含 `native`。
- RL 划分输出包括：
  - `train_object.json`
  - `test_object.json`
  - `train_cluster.json`
  - `test_cluster.json`
  - `meta.json`

## 配置策略（强制）
- 主线采用 config-first：所有 CLI 入口都必须加载 JSON 配置。
- 不允许在 Python 代码中重新构造默认值（禁止 `build_default_*` 风格）。
- 脚本默认入口配置为：
  - `configs/run_YCB_liberhand_right.json`
- 当前主线配置分组顺序为：
  - `seed`、`data`、`hand`、`sampling`、`sim_grasp`、`extforce`、`profile_object`
- 配置集命名：
  - `run_<dataset_group>_<hand>_<side>.json`，其中 dataset group 属于 `{YCB, DGN}`，hand 属于 `{liberhand, inspire, liberhand2}`，side 通常为 `{left, right}`。
- 缺失或非法的配置字段必须快速失败，并给出明确错误。

## 统一数据集格式（摘要）
- 内部样本表示应包含：
  - 每个 object-scale 一个必需输出：`grasp.h5`
  - 每个 object-scale 一个必需派生输出：`grasp.npy`
  - 每个 object-scale 一个必需失败输出：`grasp_fail.h5`
  - 每个 object-scale 一个必需失败派生输出：`grasp_fail.npy`
  - `grasp.npy` 必须由 `grasp.h5` 转换而来，且抓取数值完全一致
  - `grasp.h5` 样本 schema：
    - `object_name: str`
    - `scale: float`
    - `hand_name: str`
    - `rot_repr: "wxyz+qpos"`
    - `qpos_init: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_approach: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_prepared: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_grasp: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `qpos_squeeze: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `meta: {}`
  - `grasp_fail.h5` 样本 schema：
    - `object_name: str`
    - `scale: float`
    - `hand_name: str`
    - `rot_repr: "wxyz+qpos"`
    - `qpos_fail: [tx,ty,tz,qw,qx,qy,qz,q1...qN]`
    - `failure_stage: str`
  - 保存的 `qpos_prepared` 仍然是原始候选 pregrasp 状态
  - 回放/extforce 验证时会使用 `qpos_squeeze` 的位姿加上保存的 prepared joints 重建 pregrasp，而不是直接回放保存的 `qpos_prepared` 位姿
  - 点云单独存储，不得打包进 `grasp.npy`
  - 局部点云渲染输出（后处理，独立于抓取数组）：
    - `global_pc.npy`
    - `cam_in.npy`
    - `cam_ex_XX.npy`
    - `partial_pc_XX.npy`
    - `partial_pc_cam_XX.npy`

## 协作规则
- 修改应聚焦于用户请求的任务。
- 除非绝对必要，避免对 `src/mj_ho.py` 做结构性重写。
- 除非用户明确要求 push，否则不要主动 push 提交或分支。

## 计划索引
- 当前激活计划：`TODO.md`

## 双语同步（强制）
- `AGENTS_zh.md` 是本文件对应的中文翻译，必须与 `AGENTS.md` 保持一致。
- 每次更新 `AGENTS.md` 时，必须在同一次变更中同步更新 `AGENTS_zh.md`。

## 文档命名（强制）
- `docs/` 下自行维护的文档必须使用：
  - `YYYYMMDD_<类别>_<标题概括>.md`
- 必填类别标签：
  - `TODO`
  - `方案`
  - `调研`
- 命名顺序必须为：日期在前，类别在中，简短概括标题在后。
- 在本仓库工作流中，`docs/` 仅作为参考资料使用，不应上传到 git；除非用户明确要求，否则保持忽略并仅保留本地。
- 导出的扫描/数据产物（例如 `github_dexterous_grasp_scan.*`）不需要遵循该命名规则。

## TODO 生命周期（强制）
- 仓库根目录只保留一个激活中的计划文件：`TODO.md`。
- 替换当前 TODO 之前，先按以下命名归档到 `docs/`：
  - `docs/YYYYMMDD_TODO_<标题概括>_history.md`
- 当当前 `TODO.md` 中所有事项都被勾选完成时：
  - 按相同命名规则将该 TODO 归档到 `docs/`
  - 在仓库根目录创建下一轮新的 `TODO.md`
- 执行任务时，始终先读取根目录当前激活的 `TODO.md`；历史文件仅供参考。

## CHANGELOG 规则（强制）
- 从当前仓库状态开始，在根目录维护 `CHANGELOG.md`。
- 每次 commit 前，必须在同一次变更中更新 `CHANGELOG.md`。
- 记录保持简明，聚焦用户可见的流水线、配置、数据集或工具变更。
- 除非旧条目存在事实错误，否则不要重写历史条目；新增内容应以日期分节追加。

## Python 代码风格（简明）
- 遵循 PEP 8，并使用 `black` + `isort` 格式化。
- 为公共函数和关键数据结构添加类型标注。
- 保持入口脚本轻量，将核心逻辑放在 `src/`。
- 坚持 config-first：当配置是必需项时，不要在代码里隐藏默认值。
- 运行时日志使用 `logging`，不要使用临时 `print`。
- 快速失败，并提供明确且带上下文的错误信息。
- 对于简单直接的逻辑，优先使用直接控制流；避免不必要的回退分支、兼容分支以及宽泛的 `try/except`。
- 为核心流水线和回归问题补充/维护最小必要测试。
- 保持提交小而聚焦，提交信息清晰。

## Git 远程提醒
用户说明：remote 通过以下命令管理：
`git remote set-url origin git@github.com:HomeworldL/`

确保最终 remote URL 指向真实仓库的 SSH 路径，例如：
`git@github.com:HomeworldL/dexgrasp_sample.git`。
