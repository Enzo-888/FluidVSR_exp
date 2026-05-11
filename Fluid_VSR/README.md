# Fluid_VSR

`Fluid_VSR` 是本仓库里的内部 baseline 主框架。

它承担两类职责：

- 训练和推理内部 baseline 模型：
  `FNO`、`EDSR`、`SwinIR`、`SRNO`、`ResShift`、`ReMD`
- 作为统一评估中心：
  负责 notebook 指标计算、可视化、批量导出预测、额外频谱指标和效率统计

## 目录结构

- `main.py` / `main_ddp.py`
  训练入口。
- `template_configs/`
  各数据集、各模型的配置文件。
- `datasets/`
  数据集加载器。
- `models/`
  内部 baseline 模型实现。
- `trainers/`
  训练逻辑。
- `forecastors/`
  推理封装。`save_predictions.py` 会自动识别普通模型、`DDPM`、`ResShift` 和 `ReMG` 的 forecaster。
- `tools/`
  通用脚本，例如导出预测、算基础指标、算额外频谱指标、批量生成 bilinear 结果。
- `template_notebook/`
  统一评估 notebook。

## 训练

单卡示例：

```bash
cd /data/yc/FluidVSR/Fluid_VSR
python main.py --config template_configs/RayleighBenard/fno.yaml
```

多卡示例：

```bash
cd /data/yc/FluidVSR/Fluid_VSR
torchrun --nproc_per_node=4 main_ddp.py --config template_configs/ShallowWater/resshift.yaml
```

配置文件控制以下内容：

- 模型结构
- 数据路径
- 训练/验证/测试划分
- batch size
- 学习率
- 日志目录
- 是否 DDP

## 导出测试集预测

训练完成后，进入某个实验目录，用下面的脚本导出统一格式预测：

```bash
cd /data/yc/FluidVSR/Fluid_VSR
python tools/save_predictions.py --model_dir logs/RayleighBenard/FNO/.../
```

输出目录通常是：

```text
<model_dir>/test_predictions/
```

其中会保存：

- `pred.npz`
- `gt.npz`
- `lr.npz`
- `meta.json`

## 评估

常用 notebook 在 `template_notebook/` 下：

- `eval.ipynb`
  通用评估入口。
- `eval_RBC.ipynb`
- `eval_SW.ipynb`
- `eval_KF256.ipynb`
- `eval_ERA5.ipynb`
- `eval_RBCx8.ipynb`
- `eval_SWx8.ipynb`
- `eval_KF256x8.ipynb`
- `eval_ERA5x8.ipynb`
- `efficiency_summary.ipynb`

常用脚本在 `tools/` 下：

- `eval_metrics.py`
  基础指标。
- `eval_extra_metrics.py`
  额外频谱指标。
- `eval_spectrum_metrics_batch.py`
  对 `test_predictions` 批量计算额外频谱指标。
- `create_bilinear_test_predictions.py`
  根据已有 `lr/gt` 生成 bilinear 基线结果。

## 适用范围

这个框架是 baseline 仓库里的统一中枢。即使某个模型本身不在这里训练，最终也建议把导出的 `pred/gt/lr/meta` 结果拿回这里统一评估。
