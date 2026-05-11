# Qualitative Comparison and Zoomed View Figures

两类论文定性对比图的绘图代码。

---

## 文件说明

| 文件 | 图类型 | 示例输出 |
|------|--------|---------|
| `plot_comparison_figures.py` | LR 输入 + 模型预测行 + 误差行 | `comparison_SW.pdf`, `comparison_RBC.pdf` |
| `plot_zoom_figures.py` | LR 全图（虚线框）+ 2×3 patch 放大对比 | `paper_figures/ERA5_72_zoom.pdf`, `paper_figures/RBC_zoom.pdf` |

---

## 环境依赖

```
numpy  matplotlib  pathlib（标准库）
```

---

## 快速上手：修改路径

两个脚本顶部各有一个"修改这里"注释块，换成本地路径即可：

**`plot_comparison_figures.py`**
```python
PRED_BASE = Path('/data_new/FluidVSR_data/expriments')   # 推理结果根目录
OUT_DIR   = Path('/home/gjx/cvpr_figure1_4')             # 图片输出目录
```

**`plot_zoom_figures.py`**
```python
PRED_BASE  = Path('/data_new/FluidVSR_data/expriments')        # 推理结果根目录
OUT_DIR    = Path('/home/gjx/cvpr_figure1_4')                  # 图片输出目录
PAPER_DIR  = Path('/home/gjx/cvpr_figure1_4/paper_figures')    # zoom 图输出子目录
```

---

## 数据目录结构

脚本读取的是标准推理结果目录，每个数据集下按模型名组织：

```
{PRED_BASE}/
├── RBC/×4/test_predictions/
│   ├── srno/pred.npz  gt.npz
│   ├── edsr/
│   ├── basicvsr++/
│   ├── rvrt/
│   ├── resshift50M/
│   └── WRD_sgf_KCS+WSDF/
├── SW/×4/test_predictions/
├── KF256/×4/test_predictions/
└── ERA5_72frames/x4/test_predictions/   # 注意：ERA5_72 路径用小写 x4，需单独配置
    ├── srno/  edsr/  basicvsr++/  rvrt/
    ├── resshift50m/                      # 注意：小写 m
    └── wrd_baseline_attn/               # WRD 在 ERA5_72 下的目录名
```

每个模型目录下需有：
- `pred.npz`：预测结果，`data` 键，shape `(N×T, H, W, 1)`
- `gt.npz`：GT，相同格式

---

## 运行命令

### Comparison 图（预测 + 误差对比）

```bash
# 跑全部数据集组合
conda run -n 25cvpr python "plot_comparison_figures.py"

# 只跑某一数据集
conda run -n 25cvpr python "plot_comparison_figures.py" --dataset SW
conda run -n 25cvpr python "plot_comparison_figures.py" --dataset RBC
conda run -n 25cvpr python "plot_comparison_figures.py" --dataset ERA5_72
conda run -n 25cvpr python "plot_comparison_figures.py" --dataset KF256

# 指定序列和帧号
conda run -n 25cvpr python "plot_comparison_figures.py" --dataset RBC --seq 4 --frame 24
```

### Zoom 图（patch 放大对比）

```bash
# 单数据集
conda run -n 25cvpr python "plot_zoom_figures.py" --dataset RBC
conda run -n 25cvpr python "plot_zoom_figures.py" --dataset SW
conda run -n 25cvpr python "plot_zoom_figures.py" --dataset ERA5_72
conda run -n 25cvpr python "plot_zoom_figures.py" --dataset KF256

# 全部（含单数据集图 + 两张组合图）
conda run -n 25cvpr python "plot_zoom_figures.py" --dataset all

# 自定义帧和 patch 区域
conda run -n 25cvpr python "plot_zoom_figures.py" --dataset RBC --seq 4 --frame 24 --x0 10 --x1 62 --z0 4 --z1 56
```

---

## 各数据集默认帧

| 数据集 | seq | frame |
|--------|-----|-------|
| RBC | 4 | 24 |
| SW | 17 | 28 |
| ERA5_72 | 8 | 60 |
| KF256 | 5 | 146 |

---

## 修改模型列表

在脚本中找到 `MODELS_DEFAULT`（或 `MODELS_ERA5_72`），按格式增删：

```python
MODELS_DEFAULT = [
    ('目录名',  '显示标签',  是否高亮),   # 高亮=True 时标签变红加粗
    ('srno',    'SRNO',      False),
    ('WRD_sgf_KCS+WSDF', 'WRD (ours)', True),
]
```

顺序即为图中列的顺序（zoom 图固定 2×3，最多 6 个模型）。
