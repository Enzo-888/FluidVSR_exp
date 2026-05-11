# Plotting

这是本仓库统一的作图目录。

后续所有和论文图、补充材料图、频谱图有关的逻辑，都放在这里，不再散落到各个模型框架内部。

## 当前内容

- `rollout_latex/`
  rollout 曲线和 qualitative 图的 LaTeX / PNG / CSV 生成逻辑。

后续如果加入频谱图，建议直接作为并列子目录，例如：

- `spectrum_plots/`
- `spectral_latex/`

## 使用原则

模型训练和推理仍然在各自框架内部完成；`plotting/` 只消费已经导出的结果，例如：

- `pred.npz`
- `gt.npz`
- `lr.npz`
- `meta.json`

## 当前入口

rollout 相关说明见：

- [rollout_latex/README.md](rollout_latex/README.md)
