# RVRT 模型分析与修改说明

## 一、原版 RVRT 实验设计

### 1. 支持的任务

RVRT 是一个通用视频恢复 Transformer，支持三类任务：

- **视频超分辨率 (Video SR)**：输入低分辨率视频，输出高分辨率视频（upscale=4）
- **视频去模糊 (Video Deblurring)**：输入模糊视频，输出清晰视频（upscale=1）
- **视频去噪 (Video Denoising)**：输入带噪声视频，输出干净视频（upscale=1）

### 2. 输入输出格式

**输入**：
- 形状：`(B, T, C, H, W)` — batch × 时间帧数 × 通道 × 高 × 宽
- 通道数：**固定 3 通道 RGB**（原版硬编码）
- 数值范围：[0, 1] 归一化
- 典型尺寸：
  - SR 任务：LR 输入 64×64，HR 输出 256×256
  - Deblur/Denoise：输入输出同尺寸（如 256×256）

**输出**：
- 形状：`(B, T, C, H_out, W_out)`
- SR 任务：`H_out = H × upscale`, `W_out = W × upscale`
- 其他任务：`H_out = H`, `W_out = W`

### 3. 核心组件与数据流

```
输入 LQ (B,T,3,H,W)
    ↓
[1] SpyNet 光流估计
    - 计算相邻帧间的前向/后向光流 (T-1, 2, H, W)
    - 用于后续的帧对齐
    ↓
[2] 浅层特征提取 (RSTBWithInputConv)
    - Conv3d(3→64) + Transformer blocks
    - 输出特征 (B,T,64,H,W)
    ↓
[3] 循环特征细化 (Recurrent Feature Refinement)
    - 将视频分成多个 clip（每个 clip_size=2 帧）
    - 逐 clip 处理，前一 clip 的特征传递给下一 clip
    - 每个 clip 内部：
      a) 深层特征提取 (多个 RSTB)
      b) 引导可变形注意力 (GuidedDeformAttnPack)
         - 用光流引导注意力，对齐前一 clip 特征
         - 跨 clip 信息融合
    ↓
[4] 重建模块
    - 上采样 (Upsample)：PixelShuffle 实现 4× 放大
    - 最终卷积：Conv3d(64→3)
    ↓
输出 HQ (B,T,3,H×4,W×4)
```

**关键设计**：
- **SpyNet**：预训练光流网络，输入必须是 **3 通道 RGB**，且尺寸 ≥64px（5 层金字塔结构）
- **Recurrent 机制**：clip-by-clip 处理，平衡显存与长程依赖
- **Guided Deformable Attention**：用光流预测采样位置，再用注意力聚合特征

---

## 二、针对 RB 数据集的修改

### 修改目标
将 RVRT 适配到 **单通道物理场数据**（RayleighBenard 温度场），输入 32×32 → 输出 128×128。

### 核心修改点

#### 1. **单通道支持** (`network_rvrt.py`)

**问题**：原版硬编码 `in_channels=3`，SpyNet 只接受 RGB 输入。

**修改**：
```python
# 新增参数 in_chans（默认 3，兼容原版）
def __init__(self, ..., in_chans=3):
    self.in_chans = in_chans

    # 浅层特征提取
    self.feat_extract = RSTBWithInputConv(in_channels=in_chans, ...)

    # 最终输出层
    self.conv_last = nn.Conv3d(64, in_chans, ...)
```

**影响**：模型可处理任意通道数（1/3/N），输出通道数与输入一致。

---

#### 2. **SpyNet 单通道适配** (`compute_flow` 方法)

**问题**：SpyNet 预训练权重基于 RGB，直接输入单通道会报错。

**修改**：
```python
def compute_flow(self, lqs):
    # 如果输入不是 3 通道，复制到 3 通道
    if c != 3:
        lqs_1 = lqs_1.repeat(1, 3, 1, 1)  # (N,1,H,W) → (N,3,H,W)
        lqs_2 = lqs_2.repeat(1, 3, 1, 1)
```

**影响**：
- 单通道数据通过通道复制"伪装"成 RGB，利用预训练 SpyNet
- 光流计算仍然有效（物理场的空间梯度与 RGB 图像类似）

---

#### 3. **小尺寸输入处理** (`compute_flow` 方法)

**问题**：SpyNet 5 层金字塔要求输入 ≥64px，RB 数据 LR 只有 32×32。

**修改**：
```python
# 上采样到 64px 最小尺寸
spy_h, spy_w = max(h, 64), max(w, 64)
if spy_h != h or spy_w != w:
    lqs_1 = F.interpolate(lqs_1, size=(spy_h, spy_w), ...)
    lqs_2 = F.interpolate(lqs_2, size=(spy_h, spy_w), ...)

# 计算光流后，缩放回原尺寸
def _rescale(flow):
    flow = F.interpolate(flow, size=(h, w), ...)
    flow[:, 0] *= w / spy_w  # 缩放光流向量
    flow[:, 1] *= h / spy_h
    return flow
```

**影响**：
- 32×32 输入先插值到 64×64 计算光流，再缩放回 32×32
- 光流向量按比例调整，保证对齐精度

---

#### 4. **新增 RB 数据集** (`data/dataset_video_rb.py`)

**设计**：
- 加载 `.npz` 文件（`input` 初始场 + `output` 时序场）
- 自动 train/valid/test 分割（96/12/12）
- Min-max 归一化到 [0,1]，stats 保存到 JSON
- 数据增强：随机翻转、转置（训练时）
- 输出格式：`{'L': (T,1,h,w), 'H': (T,1,H,W)}`（与 VRT/KAIR 一致）

**关键特性**：
- `denorm_hr/denorm_lr` 方法：推理后反归一化到物理值
- `test_mode=True`：加载完整序列（100 帧），无数据增强

---

#### 5. **训练/测试脚本**

**`main_train_rvrt_rb.py`**：
- 支持单卡/多卡 DDP 训练
- Charbonnier loss（对异常值更鲁棒）
- 每 1000 iter 保存 checkpoint

**`main_test_rvrt_rb.py`**：
- 加载 checkpoint，推理测试集
- 输出 `pred.npz/gt.npz/lr.npz`（物理值）+ `meta.json`
- 格式与 `eval_rb.ipynb` 兼容

---

## 三、修改的影响与权衡

### 优势
1. **通用性提升**：支持单通道/多通道数据，不限于 RGB
2. **小尺寸友好**：32×32 输入可正常运行（原版最小 64×64）
3. **物理场适配**：光流对齐在温度场等连续场上仍然有效

### 潜在问题
1. **SpyNet 预训练偏差**：
   - 预训练基于自然图像 RGB，物理场的统计特性不同
   - 通道复制是权宜之计，理想情况应在物理场数据上微调 SpyNet

2. **小尺寸光流精度**：
   - 32→64 插值会引入伪影，光流可能不如原生 64×64 精确
   - 但实验表明影响有限（物理场空间连续性强）

3. **模型容量**：
   - 为适配 4090 24GB，缩小了模型（`embed_dims=64`, `num_blocks=[1,1,1]`）
   - 容量低于原版 REDS 配置（`embed_dims=144`, `num_blocks=[1,2,1]`）

---

## 四、与其他框架对比

| 框架 | 单通道支持 | 小尺寸处理 | 光流对齐 | 修改难度 |
|------|-----------|-----------|---------|---------|
| **RVRT** | ✅ 通道复制 | ✅ 插值到 64px | ✅ SpyNet | 中等 |
| **VRT** | ✅ 同 RVRT | ✅ 同 RVRT | ✅ SpyNet | 中等 |
| **BasicVSR++** | ✅ 修改 SPyNet | ❌ 需手动处理 | ✅ SPyNet | 较难 |
| **SDIFT** | ✅ 原生支持 | ✅ 原生支持 | ❌ 无光流 | 简单 |

**RVRT 的优势**：
- Recurrent 机制显存效率高（相比 VRT 全局注意力）
- 光流引导的可变形注意力对运动场景效果好
- 代码结构清晰，易于修改

---

## 五、使用示例

### 训练
```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 python main_train_rvrt_rb.py

# 4 卡 DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_train_rvrt_rb.py
```

### 测试
```bash
python main_test_rvrt_rb.py --checkpoint experiments/rvrt_rb/20000_G.pth
```

### 评估（使用 Fluid_VSR 框架）
在 `eval_rb.ipynb` 中配置：
```python
models = [
    {'name': 'RVRT', 'type': 'external',
     'pred_path': '/data/yc/RVRT-main/experiments/rvrt_rb/test_predictions/pred.npz'}
]
```

---

## 总结

RVRT 通过 **通道复制 + 尺寸插值** 两个简单修改，成功适配单通道小尺寸物理场数据。核心思想是"复用预训练 SpyNet"，避免从头训练光流网络。这种方案在 RB 数据集上验证有效，但对于统计特性差异更大的数据（如稀疏场、离散场），可能需要重新训练 SpyNet 或改用无光流的方法（如 SDIFT）。