# TODO

- [x] 阶段 A：手部对齐与控制接口重构
  - [x] 明确抓取对齐轴的来源与表示：手工标注、由 MuJoCo body/frame 推导、或由 mesh 几何自动估计
  - [x] 将 `_default_hand_profiles` 迁移到配置层，统一 `ctrl_joint_indices`、侧摆关节、拇指松弛关节、接触参数等 hand-specific 配置
  - [x] 设计主动自由度控制接口，明确 `qpos`、`ctrl`、可视化/回放输出之间的关系
  - [x] 评估 `method 2` 继续作为默认目标点选择方法时，对不同手型配置的依赖边界

- [x] 阶段 B：接触物理与根部稳定性策略
  - [x] 设计手根部稳定方案：超大质量、每步强制设位姿、程序内部约束/补偿，明确其对 extforce 验证的影响
  - [x] 设计摩擦与接触刚度参数方案，明确 `friction_coef`、`solimp`、`solref` 作用于手、物体或两者时的策略
  - [x] 为根部稳定与接触参数建立小规模对比实验方案，避免凭经验直接改主线

- [ ] 阶段 C：后续回到 GPU / MJWarp 版本适配
  - [ ] `global_pc.npy` 导出与 `run_multi_mjw.py` 的跳过条件对齐
  - [ ] `grasp_fail.h5` / `grasp_fail.npy` 的失败样本导出策略对齐
  - [ ] `sim_under_extforce(qpos_target, qpos_prepared)` 的新接口与回放语义对齐
  - [ ] 数据集构建、回放验证与可视化入口在 GPU 产物上完成一轮兼容性检查
