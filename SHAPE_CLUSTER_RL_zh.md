# 形状聚类与 RL 快速说明

## 运行方式
先准备物体资产：
```bash
python prepare_object_assets.py -c configs/assets_YCB.json --force --jobs 8
```

运行形状聚类：
```bash
python run_shape_cluster.py -c configs/assets_YCB.json --force
```

构建 RL 划分：
```bash
python build_dataset_splits_rl.py -c configs/assets_YCB.json --force
```

生成聚类缩略图：
```bash
python vis_shape_cluster.py -c configs/assets_YCB.json
```

## 聚类使用的数据
形状聚类使用：
- `datasets/objdata_YCB/<object>/scale120/pc_warp/global_pc.npy`
- 对点云先去中心，再归一化到单位半径
- 基于 object-level AE 特征做 KMeans

形状聚类不使用：
- `native`
- `grasp.h5`
- `grasp.npy`
- `run_warp_render.py` 产生的 partial point cloud

## 主要产物
聚类目录：
- `datasets/objdata_YCB/_meta/shape_cluster/v1_ae128_k4_seed0/`

主要文件：
- `meta.json`：配置和文件索引
- `object_features.npy`：每个物体的特征向量
- `cluster_centers.npy`：KMeans 簇中心
- `object_cluster.json`：物体到簇的映射
- `cluster_index.json`：簇到成员列表的映射
- `curriculum_index.json`：按距离从中心向外排序的 RL 课程顺序
- `train_history.json`：AE 训练 loss 历史
- `ae_state_dict.pt`：训练好的 AE 模型

RL 划分目录：
- `datasets/objdata_YCB/_meta/rl_split/v1_ae128_k4_seed0_test20_seed0/`

主要文件：
- `train_rl.json`
- `test_rl.json`
- `train_cluster_index.json`
- `test_cluster_index.json`
- `meta.json`

## 含义
- 聚类是 object-level
- RL 数据记录是 object-scale-level
- 同一个物体的所有 scale 继承同一份聚类信息
- 每个簇内部成员按到簇中心的距离排序，可直接用于课程学习
