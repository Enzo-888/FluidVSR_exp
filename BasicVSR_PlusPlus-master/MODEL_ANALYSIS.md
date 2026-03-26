# BasicVSR++ 模型架构分析与修改说明

---

## 一、原始 BasicVSR++ 模型（BasicVSR_PlusPlus/）

### 1.1 输入

| 项目 | 说明 |
|------|------|
| 张量形状 | `(n, t, 3, h, w)` — batch × 帧数 × RGB × 高 × 宽 |
| 通道数 | 3（RGB 自然图像） |
| 值域 | [0, 1]（归一化后） |
| 典型尺寸 | LR: 64×112（Vimeo90K），7帧序列 |

### 1.2 模型组件与数据流

```
输入 LQ (n, t, 3, h, w)
  │
  ▼
┌─────────────────────────────────────────┐
│ ① 特征提取 (Feature Extraction)         │
│   Conv(3→64) + 5个 ResidualBlockNoBN    │
│   每帧独立提取空间特征                    │
│   输出: (n, t, 64, h, w)                │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ ② 光流估计 (SPyNet)                     │
│   6层空间金字塔网络，预训练权重           │
│   计算相邻帧间的前向/后向光流             │
│   输出: flows_forward, flows_backward    │
│         各 (n, t-1, 2, h, w)            │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ ③ 双向传播 × 2轮 (Bidirectional Prop.)  │
│                                         │
│   第1轮: backward_1 → forward_1         │
│   第2轮: backward_2 → forward_2         │
│                                         │
│   每个分支包含:                          │
│   - SecondOrderDeformableAlignment      │
│     (二阶可变形对齐，基于光流+可学习偏移) │
│   - ResidualBlocksWithInputConv         │
│     (7个残差块做特征精炼)                │
│                                         │
│   每个分支输出: (n, t, 64, h, w)        │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ ④ 重建与上采样 (Reconstruction)          │
│                                         │
│   拼接5组特征: spatial + 4个传播分支      │
│   → (n, 320, h, w)                      │
│   → ResidualBlocks(320→64)              │
│   → PixelShuffle 2× → (n, 64, 2h, 2w)  │
│   → PixelShuffle 2× → (n, 64, 4h, 4w)  │
│   → Conv → Conv(64→3)                   │
│   → + 双线性上采样的原始输入 (残差连接)   │
└─────────────────────────────────────────┘
  │
  ▼
输出 HR (n, t, 3, 4h, 4w)
```

### 1.3 输出

| 项目 | 说明 |
|------|------|
| 张量形状 | `(n, t, 3, 4h, 4w)` — 4倍空间超分 |
| 通道数 | 3（RGB） |
| 损失函数 | Charbonnier Loss: `mean(sqrt((pred-gt)² + ε))` |

### 1.4 关键设计

- **二阶可变形对齐**: 不仅用当前帧和前一帧的光流，还利用前两帧的信息做更精确的特征对齐
- **两轮双向传播**: 信息在时间维度上前后传播两次，每帧能聚合整个序列的时序信息
- **残差学习**: 最终输出 = 网络预测的残差 + 双线性上采样的输入，降低学习难度

---

## 二、修改后的版本（BasicVSR_PlusPlus-master/）

### 2.1 修改总览

修改集中在两个方面：**模型架构适配单通道输入** 和 **新增流体数据集加载器**。其余模块（restorer、loss、flow_warp、upsample、sr_backbone_utils、basicvsr_net）完全未改动。

### 2.2 模型架构修改（basicvsr_pp.py）

共 4 处改动：

#### 改动 1：新增 `in_channels` 参数，替换所有硬编码的 `3`

```python
# 原始：所有输入/输出通道硬编码为 3
def __init__(self, mid_channels=64, num_blocks=7, ...):
    self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)   # 输入层
    self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)                           # 输出层

# 修改后：通过 in_channels 参数控制
def __init__(self, mid_channels=64, num_blocks=7, ..., in_channels=3, ...):
    self.in_channels = in_channels
    self.feat_extract = ResidualBlocksWithInputConv(in_channels, mid_channels, 5)
    self.conv_last = nn.Conv2d(64, in_channels, 3, 1, 1)
```

影响：模型不再限定 RGB 3通道，设置 `in_channels=1` 即可处理单通道物理场数据。

#### 改动 2：SPyNet 单通道兼容（compute_flow）

```python
# 新增：SPyNet 预训练于 RGB，单通道输入需复制为3通道
if c != 3:
    lqs_1 = lqs_1.repeat(1, 3, 1, 1)  # (n, 1, h, w) → (n, 3, h, w)
    lqs_2 = lqs_2.repeat(1, 3, 1, 1)
```

影响：SPyNet 权重不变，通过通道复制让单通道数据也能正常估计光流。

#### 改动 3：小分辨率输入保护（forward）

```python
# 新增：SPyNet 要求输入至少 64×64，小于则上采样
if h < 64 or w < 64:
    h_up, w_up = max(64, h), max(64, w)
    lqs_downsample = F.interpolate(..., size=(h_up, w_up), mode='bilinear')
    self._flow_scale = (h / h_up, w / w_up)

# 光流计算后，再缩放回原始特征图尺寸
if self._flow_scale is not None:
    flows = F.interpolate(flows, size=(feat_h, feat_w), mode='bilinear')
    flows[:, :, 0] *= (feat_w / fw)   # x方向缩放
    flows[:, :, 1] *= (feat_h / fh)   # y方向缩放
```

影响：RB 数据集 LR 仅 32×32，低于 SPyNet 最小要求。此改动先上采样到 64×64 送入 SPyNet，再将光流缩放回 32×32 对齐特征图。

#### 改动 4：Bug 修复（compute_flow 中 cpu_cache 分支）

```python
# 原始：mirror 模式下 flows_forward=None，直接调用 .cpu() 会崩溃
flows_forward = flows_forward.cpu()

# 修改后：加 None 保护
if flows_forward is not None:
    flows_forward = flows_forward.cpu()
```

### 2.3 新增文件

| 文件 | 用途 |
|------|------|
| `mmedit/datasets/sr_kf_dataset.py` | KF256（Kolmogorov Flow）数据集加载器 |
| `mmedit/datasets/sr_rayleighbenard_dataset.py` | RB（Rayleigh-Bénard）数据集加载器 |
| `configs/basicvsr_plusplus_kf256.py` | KF256 训练配置 |
| `configs/basicvsr_plusplus_rb.py` | RB 训练配置 |
| `tools/test_rb.py` | 推理脚本，输出反归一化后的物理值 |

### 2.4 数据集加载器设计（以 KF256 为例）

```
trajectory_*.npz  →  每个文件含 'output': (T, H, W) 单通道物理场
                     HR: (T, 256, 256)    LR: (T, 64, 64)
  │
  ▼
按轨迹索引划分 train/valid/test (80%/10%/10%)
  │
  ▼
训练集计算 min-max 归一化统计量 → 保存为 JSON → val/test 复用
  │
  ▼
归一化到 [0, 1]
  │
  ▼
训练时：随机裁剪 30 帧片段 + 空间增强(翻转/转置) + 时间镜像(30→60帧)
测试时：完整序列，无增强
  │
  ▼
输出张量: lq=(T, 1, h, w), gt=(T, 1, H, W)
```

RB 数据集加载器结构完全相同，仅文件名前缀（`sample_*.npz` vs `trajectory_*.npz`）和默认参数不同。

### 2.5 训练配置

| 参数 | KF256 | RB |
|------|-------|----|
| 输入通道 | 1 | 1 |
| 超分倍数 | 4× | 4× |
| LR 尺寸 | 64×64 | 32×32 |
| HR 尺寸 | 256×256 | 128×128 |
| 轨迹数 | 70 | 120 |
| 训练帧数 | 30帧 (镜像→60) | 30帧 (镜像→60) |
| 优化器 | Adam, lr=1e-4 | Adam, lr=1e-4 |
| SPyNet lr | ×0.25 (冻结前1000步) | ×0.25 (冻结前1000步) |
| 学习率策略 | CosineRestart, 10000步 | CosineRestart, 10000步 |
| 损失函数 | Charbonnier | Charbonnier |

---

## 三、修改影响总结

1. **通道泛化**：原始模型只能处理 RGB 图像，修改后支持任意通道数（当前用于单通道物理场）
2. **小分辨率兼容**：RB 数据 LR 仅 32×32，通过上采样-缩放机制绕过 SPyNet 的 64×64 下限
3. **SPyNet 复用**：单通道输入通过 repeat 复制为 3 通道，直接复用 RGB 预训练权重，无需重新训练光流网络
4. **Bug 修复**：修复了 mirror 模式 + cpu_cache 同时启用时的空指针崩溃
5. **数据管线**：新增两个流体数据集加载器，处理 npz 格式的物理仿真数据，包含归一化、时空增强等完整管线
