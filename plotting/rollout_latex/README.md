# Rollout LaTeX and Qualitative Figures

这个子目录负责两类图：

- rollout 曲线图
- comparison 对比图

所有输出默认都写回当前目录下，不再依赖旧的 `transfer_to_a100/rollout_latex` 目录名。

## 目录说明

- `build_rollout_latex.py`
  根据已有 `test_predictions` 结果，生成 rollout 的 CSV、LaTeX 和 manifest。
- `render_preview_pngs.py`
  把 rollout CSV 快速渲染成 PNG，便于检查。
- `build_focus_variants.py`
  生成局部放大或裁剪后的 rollout 图版本。
- `export_self_contained_rollout.py`
  导出不依赖外部 CSV 的自包含 LaTeX / PNG。
- `build_qualitative_comparison.py`
  为多数据集生成 comparison 对比图和候选样本。
- `build_qualitative_comparison_era5_72.py`
  单独处理 `ERA5_72` 的 comparison 对比图。
- `compile_all.sh`
  编译 `tex/` 下的 LaTeX 图。

## 输出内容

rollout 相关：

- `data/*_error_over_time.csv`
- `data/*_rmse_over_time.csv`
- `tex/*.tex`
- `manifest.json`

comparison 相关：

- `comparison/candidates/*.csv`
- `comparison/manifests/*.json`
- `comparison/tex/*.tex`

## 常用命令

从仓库根目录执行：

```bash
cd /data/yc/FluidVSR
python plotting/rollout_latex/build_rollout_latex.py --pred-root /path/to/test_predictions_root
```

```bash
cd /data/yc/FluidVSR
python plotting/rollout_latex/render_preview_pngs.py
```

```bash
cd /data/yc/FluidVSR
python plotting/rollout_latex/build_focus_variants.py
```

```bash
cd /data/yc/FluidVSR
python plotting/rollout_latex/export_self_contained_rollout.py
```

```bash
cd /data/yc/FluidVSR
python plotting/rollout_latex/build_qualitative_comparison.py
```

```bash
cd /data/yc/FluidVSR
bash plotting/rollout_latex/compile_all.sh
```

## 备注

- rollout 图的输入是已经整理好的 `test_predictions` 结果目录。
- 一些 comparison 脚本仍然读取项目内固定的预测结果根目录，这是因为它们本来就是论文图生成脚本，不是通用训练入口。
- 后续频谱图逻辑不要再塞进 `rollout_latex/`，直接作为 `plotting/` 下的新并列子目录维护。
