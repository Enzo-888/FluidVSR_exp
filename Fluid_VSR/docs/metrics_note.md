# PSNR / SSIM 计算方式说明

## 1. 原框架（`utils/metrics.py`）

**计算对象**：整个测试集的全部帧一次性拼成一个大 tensor，形状 `(N, H, W, C)`。

### PSNR
```
L = target.max() - target.min()     # 全测试集的动态范围，一个全局标量
m = MSE(pred, target)               # 全测试集所有像素的均值，一个全局标量
PSNR = 20·log₁₀(L) - 10·log₁₀(m)  # 整个测试集输出一个值
```
**非标准**：data_range 是全局范围而非逐帧范围，无法与文献中的 per-frame PSNR 直接比较。

### SSIM
```
卷积核：3×3 均值滤波（avg_pool2d）
```
**非标准**：标准 SSIM 使用 11×11 Gaussian 卷积核，结果与标准实现有明显差距。

---

## 2. eval_rb.ipynb（`torchmetrics functional`）

**计算对象**：每 256 帧为一个 batch 循环处理，最后加权平均。

### PSNR
```
data_range = g.max() - g.min()      # 当前 batch（256帧）内 GT 的动态范围
PSNR_batch = torchmetrics peak_signal_noise_ratio(pred, gt, data_range)
最终 PSNR  = Σ(PSNR_batch × batch_size) / N
```
**接近标准**：使用 torchmetrics 标准实现（11×11 Gaussian），data_range 按 batch 计算。
理想做法是逐帧计算再平均，但 batch 级与逐帧差异通常很小（同一数据集的帧 range 变化不大）。

### SSIM
```
卷积核：11×11 Gaussian（torchmetrics 默认）
data_range 同 PSNR，按 batch 计算
```
**标准实现**，与论文报告的 SSIM 口径一致，可与文献直接比较。

---

## 总结

| | data_range | SSIM 核 | 计算粒度 | 可与文献比较 |
|---|---|---|---|---|
| 原框架 | 全测试集全局 | 3×3 均值 | 整体一次 | ✗ |
| eval_rb.ipynb | batch 级 | 11×11 Gaussian | 逐 batch 均值 | ✓ |

**结论**：两个框架算出来的数值**不可直接比较**，eval_rb.ipynb 的结果是标准口径，用于对外报告。
