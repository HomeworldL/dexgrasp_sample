# scripts 测试脚本说明

本文档汇总当前 `scripts/` 目录下与接触参数实验相关的脚本。

## 1) `solimp_solref_experiment.py`
用途：
- 在单个 object-scale 上做 4 组手/物体软硬组合对比。
- 组合包括：
  - `hand_soft_obj_hard`
  - `hand_hard_obj_soft`
  - `hand_hard_obj_hard`
  - `hand_soft_obj_soft`

特点：
- 自动生成 case 配置到 `scripts/configs/solimp_solref_cases/`
- 支持并行运行 case（`--parallel-cases --max-workers`）
- 输出采样统计和嵌入深度统计到 `tmp/.../summary.json|csv`

示例：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-15 \
python scripts/solimp_solref_experiment.py \
  --object-scale-key YCB_013_apple__scale080 \
  --asset-dir datasets/objdata_YCB/YCB_013_apple/scale080 \
  --work-dir tmp/solimp_solref_experiment_YCB_013_apple_scale080 \
  --parallel-cases --max-workers 4
```

## 2) `solref_only_sweep.py`
用途：
- 固定 `solimp`，仅扫 `solref[0]`（timeconst）多档。
- 当前默认档位：`0.003, 0.01, 0.02, 0.05, 0.1`。

当前默认：
- `solimp=[0.9, 0.95, 0.001, 0.5, 2.0]`
- `solref=[timeconst, 1.0]`
- 手和物体使用**相同**参数（controlled comparison）。

配置与输出：
- 配置：`scripts/configs/solref_only_cases/`
- 输出：`tmp/solref_only_sweep_<object_scale>/`

示例：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-15 \
python scripts/solref_only_sweep.py
```

仅重算统计（不重跑采样）：
```bash
python scripts/solref_only_sweep.py --skip-run
```

## 3) `solimp_only_sweep.py`
用途：
- 固定 `solref`，仅扫 `solimp` 多档（硬 -> 软）。

当前默认：
- `solref=[0.003, 1.0]`
- `solimp` 五档：
  - `solimp_hard_1`: `[0.98, 0.999, 0.00005, 0.5, 2.0]`
  - `solimp_hard_2`: `[0.95, 0.995, 0.00010, 0.5, 2.0]`
  - `solimp_base`: `[0.90, 0.95, 0.00100, 0.5, 2.0]`
  - `solimp_soft_1`: `[0.70, 0.90, 0.00200, 0.5, 2.0]`
  - `solimp_soft_2`: `[0.50, 0.85, 0.00500, 0.5, 2.0]`
- 手和物体使用**相同**参数（controlled comparison）。

配置与输出：
- 配置：`scripts/configs/solimp_only_cases/`
- 输出：`tmp/solimp_only_sweep_<object_scale>/`

示例：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-15 \
python scripts/solimp_only_sweep.py
```

仅重算统计（不重跑采样）：
```bash
python scripts/solimp_only_sweep.py --skip-run
```

## 4) 关于嵌入深度统计
`solref_only_sweep.py` 与 `solimp_only_sweep.py` 当前都会写以下深度指标：
- `grasp_depth_p95_mm`, `grasp_depth_max_mm`
- `squeeze_depth_p95_mm`, `squeeze_depth_max_mm`

说明：
- 深度由 `max(0, -contact.dist)` 统计（单位 mm）。
- 统计对象为手-物体接触对。

## 5) 运行建议
- 为避免并行时资源争抢，建议始终使用：
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `taskset -c 0-15`（根据机器实际 CPU 核调整）
- 并行 case 数建议不要超过可用物理核心数。
