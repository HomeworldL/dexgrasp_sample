# dexgrasp_sample

一个基于 MuJoCo 的抓取采样仓库：面向物体 3D 数据，支持不同五指灵巧手进行离线抓取数据生成。

## 仓库功能
- 读取处理后的物体资产（YCB 风格 mesh + XML/URDF）
- 基于物体表面点云与法向生成抓取候选位姿
- 在 MuJoCo 中执行手-物体交互仿真
- 通过碰撞、接触数量、外力稳定性筛选有效抓取
- 将有效样本导出为 HDF5 数据集

本仓库目标是“抓取数据集构建”，不是在线控制器。

## 支持的手模型
- `assets/hands/liberhand/liberhand_right.xml`
- `assets/hands/liberhand2/liberhand2_right.xml`

主入口脚本：
- `run.py`：对应 `liberhand_right`
- `run_liberhand2.py`：对应 `liberhand2_right`

## 目录说明
- `src/dataset_objects.py`：物体索引、mesh/点云加载
- `src/sample.py`：抓取候选帧采样、FPS 下采样
- `src/mj_ho.py`：MuJoCo 手-物体环境与抓取验证逻辑
- `src/sq_handler.py`、`src/EMS/`：超二次曲面相关工具
- `run.py`、`run_liberhand2.py`：单物体采样
- `run_multi.py`：多物体并行采样
- `assets/`：手模型与 YCB 物体资产
- `outputs/`：输出抓取数据

## 主流程
1. 选择目标物体（`-o <object_name>`）
2. 采样物体点云与法向
3. 在表面点附近生成抓取候选帧
4. 转换为手根位姿 + 手指初始关节
5. 进行 `prepared/approach/init` 三阶段碰撞过滤
6. 仿真闭合得到 `qpos_grasp`
7. 保留接触数量足够的抓取
8. 外力扰动稳定性验证
9. 保存有效样本到 HDF5

## 安装依赖
仓库当前没有锁定依赖文件，建议自建 Python 环境后安装：

```bash
pip install numpy scipy torch trimesh open3d mujoco mujoco-viewer h5py tqdm transforms3d viser
```

## 快速使用
使用 LiberHand 对单个物体采样：

```bash
python run.py -o 002_master_chef_can
```

使用 LiberHand2 对单个物体采样：

```bash
python run_liberhand2.py -o 006_mustard_bottle
```

并行跑多个物体：

```bash
python run_multi.py -j 4 --script run.py
```

## 输出格式
每个 `(hand, object)` 的输出路径：

```text
outputs/<hand_name>/<object_name>/grasp_data.h5
```

`grasp_data.h5` 中主要字段：
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`

每一行代表一个通过验证的抓取样本（MuJoCo qpos 约定）。

## 备注
- 当前代码主体偏向数据生成与结果分析。
- `old/`、`src/backup/` 下多数文件是历史版本，不是主干流程。
- 若可视化失败，优先检查本地 MuJoCo/OpenGL 运行环境。
