# AGENTS_zh.md

## 仓库用途
基于 MuJoCo 的 3D 物体灵巧抓取采样仓库。
主线工作聚焦于离线抓取配置生成。

## 主线代码（不要漂移）
- `run_multi.py`
- `run.py`
- `run_warp_render.py`
- `src/mj_ho.py`（核心，保持稳定）
- `src/sample.py`
- `src/dataset_objects.py`

## 架构说明
- 主数据流：物体 mesh -> 表面点/法向 -> 抓取坐标系采样 -> MuJoCo 碰撞/稳定性过滤 -> HDF5 抓取状态。
- 采样后的视觉流程：每个 object-scale 完成抓取采样后，运行 `run_warp_render.py`，基于缩放后的 `coacd.obj` 渲染多视角局部点云。

## 数据集接口
- 首选数据集根目录为 `assets/objects/processed`。
- 本仓库通过符号链接接入外部 mesh 仓库：
  - `assets/objects -> /home/ccs/repositories/mesh_process/assets/objects`
- 主线默认合并数据集列表：
  - `["YCB"]`
- `DatasetObjects` 在合并后的数据集列表之上暴露全局整数 id 空间；`run.py` 通过全局 id 选择物体。

## 缩放策略
- 所有数据集统一使用配置中的固定 scale 列表（`dataset.scales`）。
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
    - 先进行无外力 settle 漂移门控
    - 再以 settle 后位姿为基准施加六个方向的外力
  - 仅保留验证通过的抓取
- 先以 HDF5 格式持久化输出（`grasp.h5`）：
  - 数据集字段：`qpos_init`、`qpos_approach`、`qpos_prepared`、`qpos_grasp`、`qpos_squeeze`
  - 主线中抓取数组存储 dtype 为 `float32`
  - 使用预分配容量并在运行时截断到最终有效长度
  - 长时间运行时周期性 flush/GC
- `grasp.h5` 完成后，加载同一批数组并转换为 `grasp.npy`（数值必须与 HDF5 完全一致）。
- `sim_dataset.py` 是基于 `train.json` / `test.json` 的数据集级回放/验证入口。
  - 使用保存的 `qpos_squeeze` 做复验
  - 支持 `--dtype float32` 或 `--dtype float64`，用于比较对精度敏感的 extforce 成功率
- 抓取输出完成后，使用 Warp 渲染物体局部点云，并保存到：
  - `datasets/<dataset_tag>/<object>/scaleXXX/<warp_render.output_subdir>/`
  - `dataset_tag` 规则：将配置文件 stem 前缀 `run_` 替换为 `graspdata_`
  - 默认子目录：`partial_pc_warp`
- 点云单独保存，不得打包进 `grasp.npy`。

## 采样流水线（GPU版本）
- 使用MJWarp
- 对于 `run_mjw.py`，保持 `output.max_cap=100`，并使用 `output.max_time_sec` 限制 extforce 阶段墙钟时间（当前默认：`180s`）。
- 如果目标是尽快为每个 object-scale 收集约 `100` 个有效抓取，优先将 `batch_size` 设在接近 `max_cap` 的范围，通常为 `128`、`256` 或 `512`。
- 在小容量采集目标下，不要默认使用过大的 `batch_size`，例如 `4096`：单次 MJWarp 步骤会过重，extforce 阶段也会明显变慢。
- `batch_size=4096` 更适合大规模抓取采集，而不适合快速达到 `max_cap=100`。
- 由于减小 `batch_size` 也会减少流入后续阶段的候选样本总数，因此 `sampling.n_points` 也应相应控制。
- 对于当前 `max_cap=100` 的目标，`sampling.n_points=2048` 是当前默认值；它能在候选覆盖率与碰撞过滤、`sim_grasp` 开销之间保持合理平衡。

## 数据集切分策略
- 使用 `build_dataset_splits.py` 扫描已完成的抓取/渲染输出，并生成 `datasets/<dataset_tag>/train.json` 与 `datasets/<dataset_tag>/test.json`。
- 按 `object_name` 做切分，而不是按 object-scale 条目切分；同一物体的所有 scale 必须留在同一个 split 中，避免泄漏。
- 默认按唯一物体数做约 `80/20` 切分，并使用配置中的 `seed` 打乱。
- 只有当 `grasp.h5`、`grasp.npy` 以及所需渲染输出都存在时，该条记录才可进入 split。
- 空的 grasp 文件在最终 manifest 使用前必须过滤掉。
- manifest 中的路径应保存为相对 dataset root 的相对路径，便于数据集整体迁移。

## 配置策略（强制）
- 主线采用 config-first：所有 CLI 入口都必须加载 JSON 配置。
- 不允许在 Python 代码中重新构造默认值（禁止 `build_default_*` 风格）。
- 脚本默认入口配置为：
  - `configs/run_YCB_liberhand.json`
- 当前主线配置分组顺序为：
  - `seed`、`dataset`、`hand`、`sampling`、`sim_grasp`、`extforce`、`output`、`warp_render`
- 配置集命名：
  - `<dataset_group>_<hand>.json`，其中 dataset group 属于 `{YCB, DGN, DGN2, HOPE}`，hand 属于 `{liberhand, inspire, liberhand2}`。
- `DGN2` 表示合并数据集：
  - `["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`
- 缺失或非法的配置字段必须快速失败，并给出明确错误。

## 统一数据集格式（摘要）
- 内部样本表示应包含：
  - 每个 object-scale 一个必需输出：`grasp.h5`
  - 每个 object-scale 一个必需派生输出：`grasp.npy`
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
  - 点云单独存储，不得打包进 `grasp.npy`
  - 局部点云渲染输出（后处理，独立于抓取数组）：
    - `cam_in.npy`
    - `cam_ex_XX.npy`
    - `partial_pc_XX.npy`
    - `partial_pc_cam_XX.npy`

## 协作规则
- 修改应聚焦于用户请求的任务。
- 除非绝对必要，避免对 `src/mj_ho.py` 做结构性重写。

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

## Python 代码风格（简明）
- 遵循 PEP 8，并使用 `black` + `isort` 格式化。
- 为公共函数和关键数据结构添加类型标注。
- 保持入口脚本轻量，将核心逻辑放在 `src/`。
- 坚持 config-first：当配置是必需项时，不要在代码里隐藏默认值。
- 运行时日志使用 `logging`，不要使用临时 `print`。
- 快速失败，并提供明确且带上下文的错误信息。
- 为核心流水线和回归问题补充/维护最小必要测试。
- 保持提交小而聚焦，提交信息清晰。

## Git 远程提醒
用户说明：remote 通过以下命令管理：
`git remote set-url origin git@github.com:HomeworldL/`

确保最终 remote URL 指向真实仓库的 SSH 路径，例如：
`git@github.com:HomeworldL/dexgrasp_sample.git`。
