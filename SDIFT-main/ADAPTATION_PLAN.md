# SDIFT → RB 视频超分辨率 改造方案

## 一、原始 SDIFT 完整流程（现状）

### 数据与任务设定

- **数据集**: Active Matter / SSF 等物理场，格式 `.h5`，shape `[N, T, H, W, D]`
  - 原始示例: `(170, 50, 1, 48, 48)`（170条序列，50帧，空间 1×48×48）
  - **D 是第三个空间维度（深度）**，对于 2D 物理场（如 Active Matter）D=1，此时框架退化为纯 2D 场处理
- **任务**: 给定稀疏随机观测点（10% 密度），重建完整时空物理场

---

### 第0步：数据预处理（`preprocessing_data.py`）

```
输入: active_matter_928_2.h5  [928, T, 1, 48, 48]
                    ↓
PDEDataPreprocesser_3D_large_data(tr_num=150)
  1. 随机排列后取前150条为训练集，其余为测试集
  2. 生成均匀坐标网格（归一化到[0,1]）:
       u_ind_uni ∈ [0,1]（H 方向，D=1 时只有一个点）
       v_ind_uni ∈ [0,1]（W 方向，48个点）
       w_ind_uni ∈ [0,1]（D 方向，48个点）
       t_ind_uni ∈ [0,1]（时间，50个点）
  3. 生成稀疏观测 mask（10% 密度，随机选点）
  4. 保存训练数据: active_matter_tr_data_150.h5（gzip压缩）
  5. 保存测试数据: active_matter_te_20.npy
  6. 保存元数据: active_matter_tr_metadata_150.npy（坐标+mask）
```

---

### 第一阶段：FTM 训练（`train_FTM.py`）

**目标**: 将物理场分解为 Tucker 形式的紧凑表示

**模型**: `Tensor_inr_3D`（`FTM_model.py`）

```
架构:
  U_net: SineLayer(1→1024) → SineLayer(1024→1024) → Dropout(0) → Linear(→R1) → Tanh
  V_net: 同上 → 输出 R2
  W_net: 同上 → 输出 R3
  (R = (1, 48, 48)，R1=1 对应 D=1 的退化维度)

待优化变量:
  basis_function（U/V/W_net 参数，所有样本和时刻共享）
  tucker_core（形状 [N, T, R1, R2, R3] = [150, 50, 1, 48, 48]，每样本每帧独立）

forward:
  training 模式: 输入 (u_ind, v_ind, w_ind) → 返回 (U矩阵, V矩阵, W矩阵)
  sampling 模式: 输入坐标 [n×3] → Kronecker 积 → A矩阵 [n × R1R2R3]

Tucker 重建:
  output = einsum(U, einsum(V, einsum(W, core)))  # 三次张量积
  output shape: [B, T, H, W, D]
```

**训练超参数**:

| 参数 | 值 |
|------|-----|
| `max_iter` | 1000 轮 |
| `batch_size` | 512 |
| `learning_rate` | 2e-4，AdamW |
| 损失函数 | RMSE（在 mask 点） + TV 正则（weight=1e-7） |
| 保存触发条件 | iter > 800 且 (loss 创历史最低 or iter%50==0) |
| 保存路径 | `./ckp/basis_*.pth` 和 `./data/core_*.mat`（仅保存最优） |

**输出**:
- `basis_am_2D_1x48x48_*.pth`：基函数网络权重
- `core_am_2D_1x48x48_*.mat`：核张量 `[150, 50, 1, 48, 48]`

---

### 第二阶段：扩散模型训练（`train_GPSD.py`）

**目标**: 学习核张量的时空分布，用于后验采样

**网络**: `Spatial_temporal_UNet`（`networks_edm.py`）

```
输入: [B, T, 1, 48, 48] 核张量（归一化到[0,1]）
      [B, T, 1] noise_labels（噪声水平σ）
      [B, T, 1] time_labels（归一化时间 t∈[0,1]）

编码器（分辨率 48→24→12）:
  每层 = UNetBlock(2D空间，FiLM调制) + Conv1d(T) 时间模块
  时间模块内部扩充 8x 通道，kernel_size=3

解码器（镜像，含跳跃连接）
输出: [B, T, 1, 48, 48] 去噪预测
```

**EDM 框架关键参数**:

| 参数 | 值 |
|------|-----|
| `sigma_min / sigma_max` | 0.002 / 80.0 |
| `sigma_data` | 0.5 |
| `P_mean / P_std` | -1.2 / 1.2 |
| `rho`（噪声调度幂） | 7.0 |
| GP 协方差 `gp_gamma` | 50（训练） |

**训练超参数**:

| 参数 | 值 |
|------|-----|
| `num_steps` | 15001 步 |
| `train_batch_size` | 16 |
| `learning_rate` | 2e-4，Adam |
| `warmup` | 5000 步线性升温 |
| `accumulation_steps` | 2（等效 batch=32） |
| `model_channels` | 40 |
| `channel_mult` | [1, 2, 2] |
| `layers_per_block` | 4 |
| `num_temporal_latent` | 8 |
| 模型保存频率 | 每 10000 步（共 1-2 次） |
| 采样可视化频率 | 每 500 步 |

**GP 驱动噪声**:
```python
K[i,j] = exp(-50·(t_i - t_j)²) + 1e-5·I
noise_gp = cholesky(K) @ noise_standard   # 时间关联噪声
x_noisy = x + σ·noise_gp
```

**输出**: `exps/gp-edm_am_*/checkpoints/ema_10000.pth` + `core_mean_std.mat`

---

### 第三阶段：后验采样重建（`message_passing_DPS.py`）

**关键超参数**:

| 参数 | 值 | 含义 |
|------|-----|------|
| `rho` | 0.01 | 观测密度（1%） |
| `MPDPS` | 0.4 | 消息传递权重 |
| `zeta` | 0.009 | 后验梯度步长 |
| `total_steps` | 20 | 采样去噪步数 |
| `gp_gamma` | 1 | 采样时 GP 长度尺度 |

**采样流程**:
```
对每条测试序列:
  1. get_te_observations → y_group（稀疏观测，按时刻分组）
  2. x_T ~ GP(0, K)（GP 协方差驱动的初始噪声）
  3. edm_post_sampler（20步 Heun 二阶 ODE）:
     每步 = Euler + 二阶修正 + (zeta/(i+1))·MPDPS梯度
  4. 反归一化: core_sample = sample·std + mean
  5. Tucker 解码: decoder(u,v,w, core_sample, basis) → 完整物理场
  6. 计算 RMSE
```

---

## 二、改造方案

### 核心数学认识

原始 SDIFT 与 VSR 任务的映射关系：

| | 原始 SDIFT | RB-VSR（本改造） |
|--|-----------|----------------|
| 观测 y | 随机稀疏点的物理场值（1-2%） | LR 网格（32×32）的像素值 |
| 观测坐标 | 随机选取的 (u,v) 连续坐标 | LR 的 32×32 固定规则坐标 |
| 观测算子 A | `basis(随机坐标)` | `basis(LR固定格点坐标)`，固定矩阵 |
| 重建目标 | 完整场（与观测网格相同分辨率） | HR 128×128 场 |
| D 维度 | 1（2D 物理场） | 1（2D 物理场，完全一致） |

**关键认识**：LR 和 HR 是同一方程在不同网格上的独立模拟结果，并非下采样关系。
因此 LR 的像素值直接作为观测 y，LR 的格点坐标直接作为观测坐标，**不需要任何 mask 操作**。

---

### 改动点1：数据预处理（`preprocessing_RB.py`，新建）

**原始做法**: 从 `.h5` 读取数据，随机稀疏采样 10% 构建 mask

**改造后**:
```
输入:
  HR: /data/yc/dataset/RB/code/rayleigh-benard-32/data1/HR/sample_*.npz
      shape: (100, 128, 128)  →  [N, T, 128, 128]
  LR: /data/yc/dataset/RB/code/rayleigh-benard-32/data1/LR/sample_*.npz
      shape: (100, 32, 32)   →  [N, T, 32, 32]

主要改动:
  1. 坐标生成（两套，分别对应 LR 和 HR）:
       hr_v_ind = linspace(0, 1, 128)   # HR 空间坐标（用于 FTM 训练+Tucker解码）
       hr_w_ind = linspace(0, 1, 128)
       lr_v_ind = linspace(0, 1, 32)    # LR 空间坐标（用于观测算子 A）
       lr_w_ind = linspace(0, 1, 32)
       u_ind    = [1.0]                  # D=1，单点，保持原框架结构

  2. 不需要 mask：LR 坐标即为观测坐标，HR 值为重建目标

  3. 数据分割: train 96 / valid 12 / test 12（按已有分割）

  4. 归一化: 使用 RB_120_norm_stats.json 的 min-max 统计量

  5. 保存格式（与原框架兼容）:
       训练数据: rb_hr_train.h5（HR 96×100×128×128×1）
       测试数据: rb_te.npy（LR+HR 12条序列）
       元数据:   rb_metadata.npy（hr坐标、lr坐标）
```

---

### 改动点2：FTM 训练（`train_FTM.py`）

**改动内容**:
```
1. Tucker 核尺寸: R=(1, 128, 128)  （匹配 HR 128×128，R1=1 对应 D=1）
   tucker_core shape: [96, 100, 1, 128, 128]

2. 训练用坐标: HR 的 128×128 规则格点
   ind_input = (u_ind_uni, hr_v_ind, hr_w_ind)

3. 训练目标: HR 像素值（128×128）拟合
   mask 不再使用随机稀疏点，而是全场监督
   （因为 FTM 训练阶段有 HR 全场作为 ground truth）

4. max_iter: 1000 → 1500（数据分辨率更大）

5. 路径: 改为 RB 数据路径

不变:
  loss_fn（RMSE + TV）、优化器（AdamW 2e-4）、保存逻辑、Tensor_inr_3D 网络
```

**为什么这样改**: Tucker 核的空间尺寸决定了解码输出的分辨率。R2=R3=128 才能解码出 128×128 的 HR 场。基函数网络输入是归一化标量坐标，与分辨率无关，一行不改。

---

### 改动点3：扩散模型训练（`train_GPSD.py`）

**改动内容**:
```
1. img_size: 48 → 128

2. channel_mult: [1,2,2] → [1,2,4,4]
   （对应分辨率 128→64→32→16，增加一级下采样保证感受野）

3. model_channels: 40 → 32
   （空间变大，减少通道控制显存）

4. train_batch_size: 16 → 8
   （128×128 核张量约 6× 显存，单卡 4090 24GB 足够）

5. num_steps: 15001 → 25001

6. save_model_iters: 10000 → 5000

7. 核张量 core.mat 路径改为 RB 输出路径

不变:
  EDM 损失、GP 协方差（gp_gamma=50）、EMA、Adam、warmup、
  Spatial_temporal_UNet 结构（只调参数）
```

**GPU 配置**: 使用空闲的 GPU 0, 5, 6, 7（4× RTX 4090 24GB）

可用多卡训练（若需要加速）:
```bash
CUDA_VISIBLE_DEVICES=0,5,6,7 torchrun --nproc_per_node=4 train_GPSD.py ...
```
或单卡即可（核张量 `[8, 100, 1, 128, 128]` ≈ 400MB，显存充裕）。

---

### 改动点4：后验采样（`message_passing_DPS.py`）

**改动内容**:
```
1. 观测构建（get_te_observations 完全重写）:
   原: 随机选 rho% 稀疏点
   改: 直接用 LR 的 32×32 格点坐标 + LR 像素值
       y = lr_frame[t]（32×32=1024个观测，每帧固定，无随机性）
       ind_conti = lr_grid_coords（固定坐标矩阵，预计算一次）

2. sample_shape: [1, T, 1, 48, 48] → [1, T, 1, 128, 128]

3. 观测算子 A 预计算:
   A = basis_function(lr_grid_coords)  # [1024 × R1R2R3]，固定，只算一次

4. 输出格式: 改为 pred.npz / gt.npz / lr.npz，shape (N_frames, H, W, 1)，物理值（已反归一化）
   指标计算不在此处做，统一交给 /data/yc/Fluid_VSR/tools/eval_metrics.py

不变:
  compute_continuous_poest 两阶段后验梯度
  edm_post_sampler ODE 积分 + 梯度注入
  GP 时间传播机制
  MPDPS, zeta 超参数（初始沿用，视效果微调）
```

**推理时间估计**（12 条测试序列，4× RTX 4090 24GB 可用，实际单卡足够）:

瓶颈分析：
- UNet 前向: 输入 `[1, 100, 1, 128, 128]`，model_channels=32，每次前向约 0.5-1s（4090）
- 每条序列 20步 ODE，每步 1-2 次 UNet 前向 + MPDPS 梯度（矩阵乘法，快）
- 每条序列约 **20-40s**
- 12 条序列总计：**约 5-10 分钟**（单卡 GPU 0）

---

### 改动点汇总

| 文件 | 改动量 | 改动性质 |
|------|--------|----------|
| `preprocessing_RB.py`（新建） | ~80行 | 读 RB npz，生成 HR/LR 坐标，保存训练/测试数据 |
| `train_FTM.py` | ~10行 | 改路径、R=(1,128,128)、全场监督 |
| `train_GPSD.py` | ~8行 | 改 img_size=128、channel_mult、batch_size |
| `message_passing_DPS.py` | ~60行 | 重写观测构建，改输出格式 |
| `FTM_model.py` | **0行** | 完全不变 |
| `networks_edm.py` | **0行** | 完全不变 |
| `utils.py` | **0行** | 完全不变 |

---

## 三、改造难度评估

### 技术难度：⭐⭐⭐☆☆（中等）

**容易的部分**：
- 核心数学和算法逻辑完全不变
- 改动集中在数据入口，不触及模型结构
- `networks_edm.py`（最复杂文件，1121行）一行不改
- RB 是 2D 场（D=1），与原框架 Active Matter 结构完全一致，无需适配 D 维度

**有挑战的部分**：

1. **显存压力**：tucker_core `[96, 100, 1, 128, 128]` ≈ 474MB（float32），FTM 阶段需要确认能否全量放入单张 4090（24GB），否则需要改成分批加载。

2. **FTM 收敛性**：核张量空间维度从 48 扩到 128，参数量 7 倍增加，可能需要更多 iter 和学习率调整。若效果不好，可尝试提高 Tucker rank 至 R=(1,192,192)。

3. **A 矩阵性质变化**：原始 A 是随机稀疏（每次采样不同），改造后 A 是固定规则格点（LR 坐标固定）。这使得后验约束更稳定，但 MPDPS 梯度步长 zeta 可能需要重调（LR 提供的信息更密集，梯度幅度不同）。

### 工程难度：⭐⭐☆☆☆（较低）

- 改动文件少，集中在两端（数据预处理 + 推理输出）
- 可分阶段验证：先单独验证 FTM 重建质量，再接扩散模型

### 总时间估计

| 阶段 | 估计时间 | GPU 配置 |
|------|---------|---------|
| 代码改造 | 1-2天 | - |
| FTM 训练（1500轮） | **4-8小时** | 单卡 GPU 0（tucker_core [96,100,1,128,128] ≈ 600MB，einsum 为瓶颈） |
| 扩散模型训练（25000步） | **6-12小时** | 4卡并行（0,5,6,7），batch=8/卡；单卡约24-48h |
| 测试推理（12条序列） | **5-10分钟** | 单卡 GPU 0 |
| 调参 | 0.5-2天 | - |

### 最大风险点

FTM 以 128×128 规则格点作为全场监督进行训练，但核张量的表达能力（Tucker rank R2=R3=128）是否足以精确重建 HR 场，是改造成功的关键。建议在正式训练前先用少量序列（5-10条）验证 FTM 的重建 RMSE，若偏高则需要调整 Tucker rank 或基函数网络深度。
