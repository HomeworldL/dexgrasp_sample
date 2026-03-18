# MJWarp 单物体多候选 GPU 并行采样迁移调研

## 背景

当前主线采样链路基于 [run.py](/home/ccs/repositories/dexgrasp_sample/run.py) 和 [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py)：

- `run.py` 面向单个 `object-scale` 执行候选抓取姿态采样与逐样本 MuJoCo 过滤。
- `run_multi.py` 通过 CPU 进程级并行把多个 `object-scale` 分摊到不同 worker。
- `src/mj_ho.py` 使用标准 MuJoCo Python API 构建 hand+object 模型，逐候选调用 `set_hand_qpos`、`is_contact`、`sim_grasp`、`sim_under_extforce`。

目标是在 `warp` 分支上新增 `run_mjw.py` 和 `src/mjw_ho.py`，后续直接对单个 `object-scale` 的大量候选抓取姿态做 GPU batched 并行仿真，而不是依赖 `run_multi.py` 的 CPU 多进程并行。

## 官方资料结论

### 1. MJWarp 的定位适合本任务

MuJoCo 官方文档明确将 MJWarp 定位为面向 NVIDIA GPU 的高吞吐并行仿真方案，而不是低延迟单步仿真。对于“单个 object-scale 上有大量候选抓取姿态，需要同构仿真筛选”的场景，这个定位是合适的。

直接含义：

- 单个候选的单步时延不一定比 CPU MuJoCo 更低。
- 但把大量候选映射为 `nworld` 个并行 worlds 后，整体吞吐有机会显著高于当前 CPU 逐候选循环。

### 2. 基本编程模型

官方教程给出的典型顺序是：

1. 用标准 MuJoCo 方式构建 `mujoco.MjModel`
2. 用 `mjw.put_model(mj_model)` 放到 GPU
3. 用 `mjw.make_data(mj_model, nworld=NWORLD)` 创建 batched data
4. 直接批量写入 `data.qpos` / `data.ctrl`
5. 调用 `mjw.forward(model, data)` 或 `mjw.step(model, data)`

这意味着：

- 模型构建层可以继续沿用当前 `MjSpec -> compile -> MjModel` 的思路。
- 运行时控制层必须改写成 batched `Data` 风格，不能继续依赖单实例 `mujoco.MjData` 的逐样本接口。

### 3. 性能优化点

官方文档明确建议对重复执行的 `mjw.step` 使用 CUDA graph capture。

对 grasp sampling 的直接意义：

- 若一轮 prepared/approach/init collision filter 或 closing grasp 采用固定步数和固定 world 数，适合 graph capture。
- 后续为了性能，`run_mjw.py` 最终应按“固定 batch size + 固定仿真子流程”组织。

### 4. 关键不兼容项

官方 feature parity 页面列出若干不支持项，和当前代码直接相关的有：

- `noslip`
- `contact override`
- 部分 CCD margin 组合

与当前仓库对照后，已有风险点如下：

- [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py) 在 `_set_friction()` 中设置了 `self.spec.option.noslip_iterations = 2`
- [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py) 依赖密集接触场景的稳定性，`noslip` 不可用后，接触质量和抓取稳定性需要重新验证
- [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py) 的 `get_contact_info()` / `is_contact()` 依赖 CPU MuJoCo 的 contact 数据读取范式，迁移时不能假设接口一致
- 当前代码里 `_set_margin()` / `_set_gap()` 虽然默认未启用，但若后续开启，必须核对是否触发 MJWarp 文档中的 CCD margin 限制

## 对现有代码的迁移判断

### 可以沿用的部分

- hand/object 合并建模的总体思路
- `run.py` 中点云采样、抓取框架采样、姿态转换、HDF5/NPY 写出
- 单个 `object-scale` 的输入输出协议
- 配置文件驱动方式

### 必须重写的部分

- `MjHO` 的运行时状态管理
- 基于 `mujoco.MjData` 的 `set_hand_qpos` / `step` / `reset`
- `is_contact()`、`get_contact_info()` 的 contact 读取逻辑
- `sim_grasp()` 中逐手指、逐 step 的 Python 循环控制方式
- `sim_under_extforce()` 的逐样本验证流程
- viewer 相关逻辑

### 建议保留但降级优先级的部分

- 可视化能力
- 精细的逐步 debug 交互

在 `warp` 分支的第一阶段，吞吐优先于 viewer。

## 推荐迁移架构

### 文件组织

- 新增 [run_mjw.py](/home/ccs/repositories/dexgrasp_sample/run_mjw.py)
- 新增 [src/mjw_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mjw_ho.py)
- 保留现有 [run.py](/home/ccs/repositories/dexgrasp_sample/run.py) 和 [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py) 作为 CPU 基线

### 运行模式

`run_mjw.py` 的目标不是“多 object-scale 并行”，而是：

- 输入一个 `object-scale`
- 生成大量候选姿态
- 将候选姿态打包成 `nworld` 批次
- 在 GPU 上并行完成 prepared / approach / init collision filter
- 对剩余候选继续并行做 closing grasp 与稳定性验证

这会把并行维度从“CPU 上按 object-scale 分进程”切换为“GPU 上按 candidate 分 worlds”。

### `src/mjw_ho.py` 的职责建议

- 保留 hand+object 模型构建入口
- 生成可被 `mjw.put_model()` 接收的 `mujoco.MjModel`
- 管理 `nworld` 维度的 batched qpos / ctrl / contact 结果
- 提供批量接口，例如：
  - `set_hand_qpos_batch`
  - `check_contact_batch`
  - `sim_grasp_batch`
  - `sim_under_extforce_batch`

注意：这不是说 API 名必须完全如此，而是接口语义应该从“单样本”切到“批样本”。

## 分阶段目标

### 第一阶段：骨架准备

- 建立 `warp` 分支
- 复制 `run.py -> run_mjw.py`
- 复制 `src/mj_ho.py -> src/mjw_ho.py`
- 写清楚调研结论与限制

本阶段不改实际仿真行为。

### 第二阶段：最小可运行 batched collision filter

目标是只替换最基础的 prepared / approach / init 三段碰撞过滤：

- 先不做完整 closing grasp
- 先不做 extforce 稳定性验证
- 先验证 batched qpos 设置、`forward/step`、contact 读取和 world mask 回传

这是整条链路里最先应该打通的里程碑，因为它直接验证 MJWarp 是否能稳定承接大量候选姿态。

### 第三阶段：batched closing grasp

- 把当前 `sim_grasp()` 的闭手过程改为 batched 版本
- 评估是否需要固定控制策略和固定步数，以适配 CUDA graph capture
- 对比 CPU 版 `qpos_grasp` 分布与 contact 数量

### 第四阶段：batched extforce validation

- 将 `object_fixed=False` 的稳定性测试迁移到 MJWarp
- 检查外力施加方式与 CPU MuJoCo 是否一致
- 对比 valid grasp 数量与速度收益

## 当前最关键的技术风险

### 1. `noslip` 不支持

这是最直接的已知差异。当前代码显式设置 `noslip_iterations`，而官方文档列出 `noslip` 为不支持项。后续迁移时需要在 `src/mjw_ho.py` 中避免继承这一配置，或在 put-model 前就做兼容处理。

### 2. Contact 数据访问范式变化

当前 pipeline 的有效性高度依赖 contact 数量、接触对和碰撞判断。若 MJWarp 暴露的 contact 结构、容量限制、回传方式与 CPU MuJoCo 不同，则 `is_contact()` / `get_contact_info()` 不能机械翻译。

### 3. 批大小与内存上限

官方文档指出 `nconmax` / `naconmax` / `njmax` 会直接影响内存和性能，并建议按环境做 trial-and-error。对 dexterous grasp 这种接触丰富的场景，world 数不能盲目拉大。

### 4. Closing grasp 和 extforce 的控制逻辑可能需要“定长化”

为了让 graph capture 和 GPU batched step 发挥优势，后续流程大概率要采用固定步数和固定批次大小。若当前 CPU 版依赖较多数据相关分支，迁移后可能需要重新组织控制流程。

## 结论

MJWarp 适合作为本仓库“单个 object-scale 上大量候选抓取姿态 GPU 并行筛选”的新后端，但不适合简单地把现有 `MjHO` 的单样本 CPU 逻辑逐行搬过去。最稳妥的路线是：

1. 保留 CPU 主线不动
2. 在 `warp` 分支上以 `run_mjw.py` / `src/mjw_ho.py` 双轨推进
3. 先打通 batched collision filter
4. 再迁移 closing grasp 与 extforce validation

## 参考资料

- MuJoCo MJWarp 文档（latest）：https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
- MuJoCo MJWarp API：https://mujoco.readthedocs.io/en/latest/mjwarp/api.html
- MuJoCo Warp tutorial notebook：https://github.com/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb

## 备注

文中关于“适合当前单 obj-scale 大批候选抓取场景”的判断，是基于官方文档对 MJWarp 高吞吐并行定位、tutorial 的 batched world 用法，以及当前 [run.py](/home/ccs/repositories/dexgrasp_sample/run.py) / [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py) 结构作出的工程推断。
