# SDIFT 模型架构与 RB 数据集适配文档

## 一、原始 SDIFT 模型架构（/data/yc/SDIFT）

### 1.1 设计目标
从**不规则稀疏观测**重建完整物理场时空演化，适用于多维连续时空物理数据。

### 1.2 三阶段流程

#### 阶段 1：Functional Tucker Model (FTM) — `train_FTM.py`
**输入**：
- 原始物理场数据（预处理后的 `.h5` 格式）
- 数据形状：`[N_samples, T, D, H, W]`（批次×时间×深度×高×宽）
- 稀疏观测 mask（训练时可选）

**处理**：
1. 将高维物理场分解为 Tucker 张量：`X ≈ Core ⊗ U ⊗ V ⊗ W`
   - `Core`: 低秩核心张量 `[N, T, R1, R2, R3]`（R1,R2,R3 << D,H,W）
   - `U, V, W`: 空间维度的连续基函数（SIREN 神经网络，输入归一化坐标 [0,1]）
2. 联合优化核心张量和基函数，最小化重建误差（RMSE + TV 正则）

**输出**：
- `core_*.mat`: 归一化核心张量（min-max 归一化到 [0,1]）
- `basis_*.pth`: 训练好的基函数网络
- 数据统计：`data_min`, `data_max`（用于反归一化）

---

#### 阶段 2：GP-based Sequential Diffusion (GPSD) — `train_GPSD.py`
**输入**：
- FTM 输出的核心张量 `core_*.mat`
- 时间索引 `t ∈ [0,1]^T`

**处理**：
1. 将核心张量序列视为时序数据，训练扩散模型（EDM 框架）
2. 噪声源：高斯过程（GP）协方差矩阵 `K(t,t') = exp(-γ(t-t')²)`，保证时间连续性
3. 去噪网络：**Temporally-Augmented U-Net**（Spatial_temporal_UNet）
   - 输入：噪声核心 `x_noisy [B, T, 1, R1, R2]` + 噪声水平 `σ` + 时间索引 `t`
   - 输出：去噪后的核心 `x_clean`
4. 训练损失：EDM 加权 L2 损失

**输出**：
- `ema_*.pth`: EMA 平滑的去噪网络权重
- `core_mean_std.mat`: 核心张量的均值和标准差（GPSD 归一化）

---

#### 阶段 3：Message-Passing DPS (MPDPS) — `message_passing_DPS.py`
**输入**：
- 稀疏观测数据 `y_group`（每个时间步的观测值列表）
- 观测坐标 `ind_conti_group`（归一化到 [0,1]）
- 观测时间索引 `y_time_ind_group`
- FTM 基函数 `basis_*.pth`
- GPSD 模型 `ema_*.pth` + `core_mean_std.mat`

**处理**：
1. **扩散采样**：从 GP 先验噪声 `x_T ~ GP(0, K)` 开始，逐步去噪
2. **后验梯度注入**（每步迭代）：
   - **Stage 1**：直接观测约束
     对有观测的时间步 `t_obs`，计算 `∇ log p(y|x) = A^T(y - Ax)`
     其中 `A = basis(观测坐标)` 是观测算子
   - **Stage 2**：时间消息传递（MPDPS 核心）
     利用 GP 回归将观测信息传播到其他时间步：
     `∇_t = Σ_{t_obs} k(t, t_obs) K^{-1}_{obs} A^T(y - Ax_agg)`
     其中 `x_agg` 是 GP 插值的核心估计
3. **解码**：最终核心 → 基函数重建 → 完整物理场

**输出**：
- 重建的完整时空场 `recon_list`
- RMSE 指标

---

### 1.3 关键组件

#### FTM_model.py — `Tensor_inr_3D`
- 3 个 SIREN 网络（`U_net`, `V_net`, `W_net`）
- 输入：归一化坐标 `(u, v, w) ∈ [0,1]³`
- 输出：基函数值 `[N_coords, R1/R2/R3]`
- 模式：
  - `training`: 返回完整基矩阵（用于 einsum 重建）
  - `sampling`: 返回单点基值（用于稀疏观测）

#### networks_edm.py — `Spatial_temporal_UNet`
- 时空 U-Net，处理 5D 张量 `[B, T, C, H, W]`
- 时间维度通过 `num_temporal_latent` 个可学习 token 编码
- 输入：噪声核心 + 噪声水平嵌入 + 时间索引嵌入
- 输出：预测的干净核心

---

## 二、RB 数据集适配版本（/data/yc/SDIFT-main）

### 2.1 适配目标
将 SDIFT 应用于 **Rayleigh-Bénard 对流视频超分辨率任务**：
- 输入：LR 32×32 低分辨率场
- 输出：HR 128×128 高分辨率场
- 时间长度：100 帧/序列

### 2.2 核心修改

#### 修改 1：数据加载（`train_FTM_RB.py`）
**原始**：从 `.h5` 文件加载预处理数据
**修改后**：
- 直接从 `.npz` 文件读取 HR 数据（`/data/yc/dataset/RB/.../HR/sample_*.npz`）
- 数据形状：`[96, 100, 128, 128]` → 扩展为 `[96, 100, 1, 128, 128]`（D=1）
- 归一化：min-max 到 [0,1]，保存 `data_min`, `data_max` 到 JSON

**影响**：
- 无需单独的预处理步骤
- 归一化统计直接保存，推理时可反归一化

---

#### 修改 2：Tucker 核心尺寸（`train_FTM_RB.py`）
**原始**：`R = (1, 48, 48)` 用于 Active Matter 数据
**修改后**：`R = (1, 128, 128)` 匹配 HR 分辨率

**影响**：
- 核心张量更大，表达能力更强（适合 128×128 高分辨率）
- 内存占用增加，但 FTM 训练仍可行（~1500 iter, 17 min）

---

#### 修改 3：监督方式（`train_FTM_RB.py`）
**原始**：稀疏 mask 监督（模拟不规则观测）
**修改后**：全场监督（`mask = all ones`）

**影响**：
- 训练更稳定，收敛更快
- 适合 VSR 任务（LR 是完整的低分辨率场，不是稀疏点）

---

#### 修改 4：GPSD 模型容量（`train_GPSD_RB.py`）
**原始**：`model_channels=40`, `num_temporal_latent=8`
**修改后**：`model_channels=16`, `num_temporal_latent=2`

**原因**：
- 单卡 4090 24GB 显存限制
- `model_channels=32` 会 OOM
- 16/2 配置峰值显存 ~23.5GB（唯一可行配置）

**影响**：
- 模型容量减小，但仍能有效学习 RB 数据的时序模式
- 训练 loss 从 1.166 降至 0.048（15000 步，4-GPU DDP）

---

#### 修改 5：DDP 支持（`train_GPSD_RB.py`）
**原始**：单 GPU 训练
**修改后**：支持 `torchrun` 多 GPU DDP

**关键修改**：
- `find_unused_parameters=True`（模型有未使用参数）
- 核心张量保持 CPU，batch 在训练循环内 `.to(device)`
- EMA 跟踪 `model.module`（非 DDP wrapper）

**影响**：
- 4-GPU 训练加速 ~3.5×（15000 步从 ~8h 降至 ~2.2h）

---

#### 修改 6：观测算子（`inference_RB.py`）
**原始**：随机稀疏点观测（1% 采样率）
**修改后**：LR 32×32 网格观测（1024 个固定点）

**处理流程**：
1. 预计算 LR 坐标网格：`(u=1.0, v∈[0,1]^32, w∈[0,1]^32)`
2. 每帧观测：`y_t = LR[t, 0, :, :].ravel()` (1024 维向量)
3. 观测算子：`A = basis(LR_coords)` (1024×16384 矩阵，预计算一次)

**影响**：
- 观测密度更高（1024 vs 原始 ~500 点）
- 观测结构化（网格 vs 随机点），更适合 VSR 任务

---

#### 修改 7：输出格式（`inference_RB.py`）
**原始**：保存为 `.mat` 文件
**修改后**：保存为 `.npz` 文件（与 Fluid_VSR 框架兼容）

**输出文件**：
- `pred.npz`: `[N_frames, H, W, 1]` 预测 HR（物理值，已反归一化）
- `gt.npz`: `[N_frames, H, W, 1]` 真实 HR
- `lr.npz`: `[N_frames, H_lr, W_lr, 1]` 输入 LR
- `meta.json`: 元数据（帧数、序列数、归一化统计、checkpoint 路径）

**影响**：
- 可直接用 `/data/yc/Fluid_VSR/tools/eval_metrics.py` 计算指标
- 可直接加载到 `eval_rb.ipynb` 可视化

---

#### 修改 8：Bug 修复（`inference_RB.py`）
**问题**：`torch.kron` 在某些 PyTorch 版本返回 1D 张量
**修复**：改为显式外积 `coeff.unsqueeze(-1) * post.view(1,1,-1)`

**影响**：
- 避免形状不匹配错误
- 保证 MPDPS 消息传递正确执行

---

### 2.3 新增文件

| 文件 | 功能 |
|------|------|
| `train_FTM_RB.py` | RB 数据集 FTM 训练（直接读 npz，全场监督） |
| `train_GPSD_RB.py` | RB 数据集 GPSD 训练（支持 DDP，小模型配置） |
| `inference_RB.py` | RB 数据集推理（LR 网格观测，输出 npz 格式） |
| `validate_pipeline.py` | 端到端流程验证脚本 |
| `mem_test.py` | 显存测试脚本 |

---

## 三、完整数据流

### 3.1 训练阶段
```
HR npz [96,100,128,128]
    ↓ train_FTM_RB.py
core [96,100,1,128,128] + basis + norm_stats
    ↓ train_GPSD_RB.py
GPSD model (ema_*.pth) + core_mean_std.mat
```

### 3.2 推理阶段
```
LR npz [12,100,32,32]
    ↓ 归一化 (用 FTM norm_stats)
LR_norm [12,100,1,32,32]
    ↓ build_lr_observations
y_group (每帧 1024 点) + LR_coords
    ↓ MPDPS 采样 (GPSD + 后验梯度)
core_pred [1,100,1,128,128]
    ↓ 解码 (basis 重建)
HR_pred [100,128,128]
    ↓ 反归一化 + 保存
pred.npz [100,128,128,1]
```

---

## 四、关键差异总结

| 维度 | 原始 SDIFT | RB 适配版 |
|------|-----------|----------|
| **任务** | 稀疏观测重建 | 视频超分辨率 |
| **观测类型** | 随机稀疏点（1%） | LR 网格（32×32） |
| **监督方式** | 稀疏 mask | 全场监督 |
| **核心尺寸** | (1,48,48) | (1,128,128) |
| **模型容量** | mc=40, ntl=8 | mc=16, ntl=2 |
| **训练方式** | 单 GPU | 支持 DDP |
| **输出格式** | .mat | .npz (Fluid_VSR 兼容) |
| **数据加载** | .h5 预处理 | 直接读 .npz |

---

## 五、使用示例

### 5.1 训练
```bash
# 1. FTM 训练
conda run -n cvpr python train_FTM_RB.py

# 2. GPSD 训练（4-GPU DDP）
CUDA_VISIBLE_DEVICES=0,5,6,7 torchrun --nproc_per_node=4 train_GPSD_RB.py \
    --core_path ./data/core_rb_1x128x128_2026_03_16_08.mat

# 3. 推理
CUDA_VISIBLE_DEVICES=0 python inference_RB.py \
    --basis_path ./ckp/basis_rb_1x128x128_2026_03_16_08.pth \
    --model_path ./exps/gp-edm_rb_20260316-0833/checkpoints/ema_14999.pth \
    --core_mean_std_path ./exps/gp-edm_rb_20260316-0833/core_mean_std.mat \
    --norm_stats_path ./data/norm_stats_rb_2026_03_16_08.json \
    --output_dir ./output_rb \
    --model_channels 16
```

### 5.2 评估
```bash
# 使用 Fluid_VSR 框架评估
cd /data/yc/Fluid_VSR
# 在 eval_rb.ipynb 中添加 SDIFT 模型配置（type='external'）
```

---

## 六、性能指标（RB 测试集）

| 模型 | PSNR | SSIM | 训练时间 |
|------|------|------|---------|
| FTM | - | - | ~17 min (1500 iter) |
| GPSD | - | - | ~2.2 h (15000 步, 4-GPU) |
| SDIFT (完整) | 待测 | 待测 | - |

---

**文档版本**：v1.0
**最后更新**：2026-03-17
**作者**：基于 /data/yc/SDIFT 和 /data/yc/SDIFT-main 代码分析生成
