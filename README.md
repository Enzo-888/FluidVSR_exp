# FluidVSR Baseline Repository

这是项目的官方 baseline 仓库。

它只负责 baseline 模型、外部对比模型、统一评估和作图逻辑；自研模型 `WRD` 的正式代码仓库是分开维护的，不放在这里。

## 当前组织方式

本仓库现在固定分成 `5` 个模型框架和 `1` 个作图目录：

- `Fluid_VSR/`
  内部 baseline 主框架。负责 `FNO`、`EDSR`、`SwinIR`、`SRNO`、`ResShift`、`ReMD` 的训练、推理、统一评估 notebook 和若干通用工具。
- `BasicVSR_PlusPlus-master/`
  外部 `BasicVSR++` 基线的本地适配版本。
- `KAIR-master/`
  外部 `KAIR` 框架；在本项目里只使用其中的 `VRT` 路线。
- `RVRT-main/`
  外部 `RVRT` 基线的本地适配版本。
- `SDIFT-main/`
  外部 `SDIFT` 基线的本地适配版本，已经按本项目的配对超分任务做过修改。
- `plotting/`
  统一作图目录。当前已经整合 `rollout_latex/`；后续频谱图等绘图逻辑也放在这里。

## 各子目录入口

- [Fluid_VSR/README.md](Fluid_VSR/README.md)
  内部 baseline 主框架的训练、导出预测、统一评估方法。
- [BasicVSR_PlusPlus-master/README.md](BasicVSR_PlusPlus-master/README.md)
  `BasicVSR++` 在本项目中的配置、训练和推理入口。
- [KAIR-master/README.md](KAIR-master/README.md)
  `VRT` 在本项目中的配置、训练和推理入口。
- [RVRT-main/README.md](RVRT-main/README.md)
  `RVRT` 在本项目中的训练和推理入口。
- [SDIFT-main/README.md](SDIFT-main/README.md)
  `SDIFT` 的三阶段流程，以及当前超分版本应该怎么跑。
- [plotting/README.md](plotting/README.md)
  统一作图目录的说明，以及 `rollout_latex` 的使用方法。

## 统一结果格式

不管模型在哪个框架里训练，最后都尽量导出同样的测试结果格式：

- `pred.npz`
- `gt.npz`
- `lr.npz`
- `meta.json`

这样后续就能统一回到 `Fluid_VSR/` 下的 notebook 或 tools 做评估。

## 统一工作流

1. 在各自框架里训练模型。
2. 用各自框架的测试脚本导出 `test_predictions`。
3. 把预测结果整理成 `pred/gt/lr/meta` 四件套。
4. 回到 `Fluid_VSR/template_notebook/` 或 `Fluid_VSR/tools/` 做指标、可视化和效率统计。
5. 如果需要论文图，再进入 `plotting/` 里的脚本生成 rollout / qualitative / LaTeX 图。

## 备注

- 本仓库关注 baseline，不包含 `WRD` 训练代码。
- 某些外部框架在本仓库里只保留了本项目实际使用到的入口，不追求保留原始仓库的全部功能。
- 如果某个模型在某个数据集上的实验是在别的机器上完成的，这里可能只保留推理、评估和作图所需脚本，而不一定保留完整训练产物。
