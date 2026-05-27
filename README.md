# 运行流程

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

数据集会下载到 `dataset/BOSSbase_1.01/`；如已存在则跳过。

```bash
make data
```

参数（可选）：`N_TRAIN=7000 N_TEST=3000`（默认）。

## 完整流程

```bash
# 小规模验证（350 训练 / 150 测试，db4 @ 1.0 bpp，~5 分钟）
make verify

# 大规模实验（7000 训练 / 3000 测试，所有小波 × 所有嵌入率，耗时较长）
make train-all N_TRAIN=7000
```

## 分步执行

```bash
# 1. Spearman 相关性分析 → output/figures/spearman_*.png
make spearman WAVELET=db4 RATE=2.0 N_TRAIN=350

# 2. 特征提取 → output/features/*.npy
make features WAVELET=db4 RATE=1.0 N_TRAIN=350 N_TEST=150

# 3. 训练 → model/*.pt / model/*.pkl
make train-wdcnn WAVELET=db4 RATE=1.0 N_TRAIN=350
make train-mlp   WAVELET=db4 RATE=1.0 N_TRAIN=350
make train-svm   WAVELET=db4 RATE=1.0 N_TRAIN=350

# 4. 测试 → output/results/*.csv
make test-all WAVELET=db4 RATE=1.0 N_TRAIN=350 N_TEST=150

# 5. 分析 → output/figures/roc_*.png, auc_comparison_*.png + output/results/summary.csv
make analyze N_TRAIN=350
```

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `WAVELET` | `db4` | 小波基（`haar` / `db4`） |
| `RATE` | `1.0` | 嵌入率 bpp（`1.0` / `1.5` / `2.0`） |
| `N_TRAIN` | `350` | 训练图像数 |
| `N_TEST` | `150` | 测试图像数 |

每个 `src/` 下的脚本也可以通过 `--help` 查看完整参数：

```bash
cd src && python train_wdcnn.py --help
```

## 输出目录

| 目录 | 内容 |
|---|---|
| `model/` | 训练好的模型权重（`.pt` / `.pkl`） |
| `output/features/` | 缓存的特征数组（`.npy`） |
| `output/results/` | 测试结果 CSV + `summary.csv` |
| `output/figures/` | ROC 曲线、Spearman 热力图、AUC 对比图 |

## 清理

```bash
make clean   # 清空 model/ 和 output/
```
