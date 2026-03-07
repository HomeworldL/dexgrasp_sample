# dexgrasp_sample

基于 MuJoCo 的灵巧手抓取构型采样仓库（面向物体 3D 数据）。

## 主流程代码
- `run.py`
- `run_multi.py`
- `src/mj_ho.py`
- `src/sample.py`
- `src/dataset_objects.py`

## 功能流程
1. 加载物体数据
2. 采样表面点和法向
3. 生成抓取候选位姿
4. 通过 MuJoCo 做碰撞/闭合/稳定性筛选
5. 输出有效抓取状态

## 数据接口（外部链接）
`DatasetObjects` 已支持通用目录结构，不再仅限 YCB。

默认数据根目录优先级：
1. `DEXGRASP_OBJECTS_ROOT`
2. `assets/objects/processed`

其中 `assets/objects` 采用软链接方式接入外部数据目录，避免大规模拷贝。

## 配置驱动入口
所有 CLI 入口均通过 JSON 配置驱动，不再在程序内构造默认配置：
- `run.py`
- `run_multi.py`
- `vis_obj.py`
- `vis_ho.py`
- `run_liberhand2.py`

所有入口默认配置：
- `configs/run_YCB_liberhand.json`

配置矩阵（数据集 x 手模型）：
- `run_DGN2_liberhand.json`、`run_YCB_liberhand.json`、`run_HOPE_liberhand.json`
- `run_DGN2_liberhand2.json`、`run_YCB_liberhand2.json`、`run_HOPE_liberhand2.json`
- `run_DGN2_inspire.json`、`run_YCB_inspire.json`、`run_HOPE_inspire.json`

数据集语义：
- `DGN2 = ["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]`

若配置缺失或字段不合法，程序会直接报错退出。

## 可视化脚本位置
可视化脚本已与主流程分离，统一放在：
- `tools/visualization/`

## 快速运行
```bash
python run.py -c configs/run_YCB_liberhand.json -o 002_master_chef_can
python run_multi.py -c configs/run_DGN2_liberhand.json -j 4 --script run.py
python vis_obj.py -c configs/run_YCB_liberhand.json -i 0
python vis_ho.py -c configs/run_YCB_liberhand2.json -i 0
```

## 测试
```bash
pytest -q
```

## 依赖
见 `requirements.txt`。
