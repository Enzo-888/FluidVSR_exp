# Error Energy Spectrum Figures

逐帧误差能量谱图的绘图代码，对应论文中形如下图的曲线图：
- X 轴：波数 k（FREQUENCY）
- Y 轴：误差能量谱（ERROR ENERGY，log scale）
- 多模型曲线对比，含 LR Nyquist 和中频竖线分隔三个频段

---

## 文件说明

| 文件 | 作用 | 输出 |
|------|------|------|
| `plot_error_spectrum_perframe.py` | RBC 全量逐帧谱（12 seq × 100 帧 = 1200 张） | `OUT_ROOT/seq_XX/frame_XXX.png` |
| `plot_error_spectrum_perframe_general.py` | ERA5 / KF256 / SW 全量逐帧谱 | `OUT_BASE/spec_per_frame/{DS}/seq_XX/frame_XXX.png` |
| `gen_tex_spectra.py` | 论文选定帧：PNG 预览 + pgfplots `.tex` 文件 | `OUT_DIR/{DS}/{name}.png` + `OUT_DIR/{name}.tex` |

> **典型用法**：先跑前两个脚本生成全量逐帧图，浏览后挑选感兴趣的帧，再在 `gen_tex_spectra.py` 的 `FRAMES` 列表中配置，生成论文质量的 TeX 图。

---

## 环境依赖

```
numpy  matplotlib  scipy
```

---

## 快速上手：修改路径

每个脚本顶部都有"修改这里"注释，换成本地路径即可。

**`plot_error_spectrum_perframe.py`（RBC 专用）**
```python
PRED_ROOT = Path('/data_new/FluidVSR_data/expriments/RBC/×4/test_predictions')  # RBC 推理结果目录
OUT_ROOT  = Path('/home/gjx/VSR_temporal_metrics/figures/spec_per_frame/RBC')   # 图片输出目录
```

**`plot_error_spectrum_perframe_general.py`（ERA5 / KF256 / SW）**
```python
PRED_BASE = Path('/data_new/FluidVSR_data/expriments')       # 推理结果根目录
OUT_BASE  = Path('/home/gjx/VSR_temporal_metrics/figures')   # 图片输出根目录
```

**`gen_tex_spectra.py`（论文选定帧）**
```python
PRED_BASE    = Path('/data_new/FluidVSR_data/expriments')                                    # 推理结果根目录
PRED_ERA5_72 = Path('/data_new/FluidVSR_data/expriments/ERA5_72frames/x4/test_predictions') # ERA5_72 单独路径
OUT_DIR      = Path('/home/gjx/VSR_temporal_metrics/figures/selected_frames')                # 输出目录
```

---

## 数据目录结构

```
{PRED_BASE}/
├── RBC/×4/test_predictions/
│   ├── basicvsr++/pred.npz  gt.npz
│   ├── srno/
│   ├── edsr/
│   ├── rvrt/
│   ├── resshift50M/
│   ├── WRD/
│   └── WRD_sgf_KCS+WSDF/
├── ERA5/×4/test_predictions/       # ERA5（24帧/seq）
├── KF256/×4/test_predictions/      # KF256（180帧/seq）
├── SW/×4/test_predictions/         # SW（72帧/seq）
└── ERA5_72frames/x4/test_predictions/   # ERA5_72（72帧/seq，注意小写 x4）
    ├── resshift50m/                      # 注意：小写 m
    └── wrd_baseline_attn/               # WRD 在 ERA5_72 下的目录名
```

每个模型目录下需有：
- `pred.npz`：预测结果，`data` 键，shape `(N×T, H, W, 1)`
- `gt.npz`：GT，相同格式

---

## 运行命令

### 全量逐帧谱（用于选帧浏览）

```bash
# RBC（12 seq × 100 帧，输出 1200 张，耗时较长）
conda run -n 25cvpr python "plot_error_spectrum_perframe.py"

# ERA5（单个数据集）
conda run -n 25cvpr python "plot_error_spectrum_perframe_general.py" --dataset ERA5

# KF256
conda run -n 25cvpr python "plot_error_spectrum_perframe_general.py" --dataset KF256

# SW
conda run -n 25cvpr python "plot_error_spectrum_perframe_general.py" --dataset SW

# 全部串行（ERA5 + KF256 + SW）
conda run -n 25cvpr python "plot_error_spectrum_perframe_general.py" --dataset all
```

### 论文选定帧（PNG 预览 + pgfplots TeX）

```bash
# 全部 FRAMES 列表中的帧
conda run -n 25cvpr python "gen_tex_spectra.py"

# 只跑某个数据集
conda run -n 25cvpr python "gen_tex_spectra.py" --dataset ERA5_72
conda run -n 25cvpr python "gen_tex_spectra.py" --dataset RBC
```

---

## 各数据集参数

| 数据集 | NX | 帧数/seq | seq 数 | LR Nyquist | 中频边界 |
|--------|----|---------:|-------:|-----------:|---------:|
| RBC | 128 | 100 | 12 | k=16 | k=32 |
| ERA5 | 256 | 24 | 6 | k=32 | k=64 |
| ERA5_72 | 256 | 72 | 10 | k=32 | k=64 |
| KF256 | 256 | 180 | 7 | k=32 | k=64 |
| SW | 128 | 72 | 20 | k=16 | k=32 |

---

## 修改选定帧（gen_tex_spectra.py）

在 `FRAMES` 列表中增删条目，每条指定数据集、seq、帧号和 patch 参数：

```python
FRAMES = [
    dict(name='ERA5_seq08_frame060',  dataset='ERA5_72', seq_i=8,  t=60,
         nx=256, fps=72,  k_max=110, lr_nyq=32, mid=64,
         pred_root=PRED_ERA5_72, models=MODELS_ERA5_72),
    dict(name='RBC_seq01_frame044',   dataset='RBC',   seq_i=1,  t=44,
         nx=128, fps=100, k_max=56,  lr_nyq=16, mid=32),
    # 新增帧：
    dict(name='RBC_seq04_frame081',   dataset='RBC',   seq_i=4,  t=81,
         nx=128, fps=100, k_max=56,  lr_nyq=16, mid=32),
]
```

`name` 决定输出文件名（`{name}.tex` / `{DS}/{name}.png`）。

---

## 修改模型列表

在 `MODELS`（或 `MODELS_ERA5_72`）中按格式增删：

```python
MODELS = [
    ('目录名',        'hex色',  'linestyle',   'linewidth',   '图例标签'),
    ('basicvsr++',   '1f77b4', 'solid',       'very thick',  'BasicVSR++'),
    ('WRD_sgf_KCS+WSDF', '7f2704', 'dashdotted', 'very thick', r'WRD$_{\rm sgf}$+KCS+WSDF'),
]
```
