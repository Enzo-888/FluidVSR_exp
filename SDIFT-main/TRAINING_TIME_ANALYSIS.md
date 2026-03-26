# SDIFT 全参数手册

> 覆盖所有影响**模型表现 / 显存占用 / 训练时间 / 推理时间**的参数。
> 分三个阶段：FTM → GPSD → Inference。

---

## 重点参数深度解析

### `accumulation_steps` 对本实验的具体影响

**背景**：GPSD 的训练数据是 96 条训练序列对应的 Tucker 核张量，每条序列是一个 `[100, 1, 128, 128]` 的时序张量。扩散模型要学习的是"这类时序核张量的分布"。

**`train_batch_size=1`** 意味着：每次 mini-batch 从 96 条序列里随机抽 1 条喂给模型。

**梯度累积的作用**：
```
accumulation_steps=16：
  每次权重更新前，连续随机抽 16 条序列分别算梯度，把梯度平均后再更新。
  相当于模型每步"见过" 16 条不同序列的数据，梯度方向更准确。

accumulation_steps=1：
  每抽 1 条序列就立刻更新权重，梯度只来自这 1 条序列，方向更"随机"。
```

**对本实验的实际影响**：

| | accum=16 | accum=4 | accum=1 |
|--|---------|---------|---------|
| 每步见几条序列 | 16条 | 4条 | 1条 |
| 梯度噪声 | 低 | 中 | 高 |
| 对模型质量影响 | 原始设计 | 轻微降低（baseline可接受） | 有一定影响，可能震荡 |
| 耗时（相对accum=1） | ×16 | ×4 | ×1 |

**关键问题**：96条序列，accum=1 时 15000 步 = 模型平均见每条序列 **156次**，训练数据量本身不少。扩散模型对梯度噪声有一定容忍度（加噪本身就是随机的）。accum=4 是合理的 baseline 折中——比 accum=1 稳定，比 accum=16 快 4 倍。

---

### `num_temporal_latent` 与时序连续性的具体联系

**背景**：RayleighBenard 流场是时序连续的，100帧之间有强时间相关性。GPSD 用 GP 核 `K[i,j]=exp(-50*(t_i-t_j)²)` 生成时序相关噪声来训练扩散模型，要求 UNet 能理解并还原这种时序结构。

**`num_temporal_latent` 的作用位置**：UNet 每个空间卷积层之后，紧接着做一个 1D 时序卷积：
```
输入：[B×H×W, C, T]          ← 每个空间位置的 T=100 帧特征
        ↓  Conv1d 升维
中间：[B×H×W, num_temporal_latent×C, T]   ← 扩展时序表达空间
        ↓  Conv1d 降回
输出：[B×H×W, C, T]          ← 时序信息混合后的特征
```

**`num_temporal_latent` 控制的是这个时序混合有多"复杂"**：

| 值 | 时序建模能力 | 类比 | 适用场景 |
|----|------------|------|---------|
| 1 | 只能学简单线性滤波（时序平滑） | 相当于一个普通 FIR 滤波器 | 时序变化极简单 |
| 2 | 能学 2 种时序模式（如趋势+振荡） | 类似 2阶滤波器 | baseline |
| 8 | 能学 8 种时序模式，捕捉复杂时序结构 | 原始设计，针对更复杂数据 | OOM |

**对 RayleighBenard 的具体含义**：
- RB 流场有对流卷（convection roll）结构，在 100 帧内会演化，但变化相对规律
- `num_temporal_latent=2`：能捕捉基本的时序平滑性和简单周期性，对 baseline 够用
- `num_temporal_latent=8`：原始设计针对更复杂的稀疏重建场景，对 RB 这种相对规律的流场是过设计，且 128×128 空间分辨率下直接 OOM

**结论**：`num_temporal_latent=2` 对 RB baseline 是合理的——能建模基本时序连续性，不会 OOM。若最终效果差再考虑升到 4（需要更大显存）。

---

## 概念速查

| 术语 | 含义 |
|------|------|
| `max_iter` | FTM 训练迭代次数 |
| `num_steps` | GPSD **训练**步数（权重更新次数），与推理无关 |
| `accumulation_steps` | 每次权重更新前先跑 N 次 forward/backward，时间 ×N，梯度更稳定 |
| `total_steps` | **Inference** 时扩散模型的 ODE 去噪步数，才是"推理步数" |
| `model_channels` | UNet 基础通道数 |
| `num_temporal_latent` | 时序 Conv1d 中间通道倍数，控制时序建模复杂度，显存杀手 |
| Tucker rank R | FTM 的张量分解秩 R=(R1,R2,R3) |

---

## 阶段一：FTM 训练 (`train_FTM_RB.py`)

### 任务
用 Tucker 分解 + SIREN 基函数网络拟合 96 条 HR 训练序列，学出：
- 共享基函数网络（`basis_function`，参数量约 6.3M）
- 每条序列每帧的核张量（`tucker_core`，形状 `[N_train, T, R1, R2, R3]`）

---

### 参数表

#### 影响模型表现的参数

| 参数 | 默认值 | 含义 | 调大影响 | 调小影响 |
|------|--------|------|---------|---------|
| `--R` | `1 128 128` | Tucker 秩 R=(R1,R2,R3)。R1=1因为RB是2D场（D=1），R2/R3 控制空间表达能力 | 表达能力更强，但几乎不必要（R2=R3=128已等于空间分辨率，增大无意义） | 表达能力下降，重建 RMSE 升高 |
| `--learning_rate` | `2e-4` | AdamW 学习率 | 收敛快但可能震荡 | 收敛慢 |
| `--max_iter` | `1500` | 训练迭代数 | 拟合更精确，收益递减 | 拟合不充分 |
| `omega` | `20`（硬编码） | SIREN 激活函数频率，控制基函数能表示的空间频率范围 | 可表示更高频细节 | 过于平滑 |
| `mid_channel` | `1024`（硬编码） | 基函数网络每层神经元数 | 表达能力更强 | 表达能力弱 |

#### 影响显存的参数

| 参数 | 显存估算 | 说明 |
|------|---------|------|
| `--R 1 128 128` | tucker_core：`96×100×1×128×128`×4B = **630 MB**；AdamW 动量：×3 = **1.9 GB** | 主要显存占用 |
| `--batch_size 96` | 一次加载全部 96 条序列的数据 | HR 数据：`96×100×1×128×128`×4B = 630 MB |
| 基函数网络参数 | 约 **25 MB**（3个子网，每个 1→1024→1024→R） | 可忽略 |
| **总显存** | **约 3–4 GB** | FTM 阶段不是显存瓶颈 |

#### 影响训练时间的参数

| 参数 | 默认值 | 时间估算 | 建议（baseline） |
|------|--------|---------|----------------|
| `--max_iter` | `1500` | ~15 min（96条序列） | **500**（已够，validate实测RMSE=0.0079） |

> **validate 实测**：5条序列，500 iter → 0.1 min。96条约 ×19 数据量 → 500 iter ≈ 2 min，1500 iter ≈ **5–15 min**。

---

## 阶段二：GPSD 训练 (`train_GPSD_RB.py`)

### 任务
在 Tucker 核张量空间训练扩散模型（EDM），学习核张量随时间的分布。
网络：`Spatial_temporal_UNet`，每层先做 2D 空间卷积，再做 1D 时序卷积。

---

### 参数表

#### 影响模型表现的参数

| 参数 | 默认值 | 含义 | 调大影响 | 调小影响 |
|------|--------|------|---------|---------|
| `--num_steps` | `25001` | 总训练步数（权重更新次数） | 训练更充分 | 欠拟合 |
| `--model_channels` | `32` | UNet 基础通道数，所有分辨率的通道数均为 `model_channels × channel_mult[i]` | 模型容量更大，表达能力更强 | 模型太小，拟合能力差 |
| `--channel_mult` | `1 2 4 4` | 各分辨率级别的通道倍数。当前4个级别：128×128→64×64→32×32→16×16，通道数分别为 32/64/128/128 | 增加层数或通道，容量更大 | 过浅，难以学习复杂分布 |
| `--num_blocks` | `4` | 每个分辨率级别的残差块数量 | 每级更深 | 太浅 |
| `--learning_rate` | `2e-4` | Adam 学习率 | 收敛快，但可能不稳定 | 收敛慢 |
| `--warmup` | `5000` | 前 warmup 步 lr 从 0 线性升到最大值。**注意：若 num_steps=5000，则整个训练都在 warmup，lr 永远到不了最大值** | lr 上升更平稳 | 初期 lr 过大，训练不稳定 |
| `--gt_guide_type` | `l2` | 训练 loss 类型（l2 或 l1） | — | — |
| `--sigma_min/max` | `0.002 / 80.0` | EDM 噪声调度范围 | 不建议修改 | — |
| `--sigma_data` | `0.5` | 数据标准差假设，影响 EDM 预条件（c_skip/c_out/c_in） | 不建议修改 | — |
| `--P_mean / P_std` | `-1.2 / 1.2` | 训练时采样噪声级别的 log-normal 分布参数 | 不建议修改 | — |

#### 影响显存的参数 ← 重点

时序 Conv1d 是显存瓶颈。每个 UNet 层后做：
```
[B×H×W, C, T]  →  [B×H×W, num_temporal_latent×C, T]  →  [B×H×W, C, T]
```

| 参数 | 默认值 | 显存公式 | 实例（128×128层，B=1，T=100） |
|------|--------|---------|---------------------------|
| `--num_temporal_latent` | `8`→已改`2` | B×H×W × num_temporal_latent×C × T × 4B | **2**：1×16384×2×32×100×4 = 419 MB；**8**：1×16384×8×32×100×4 = **1.68 GB**（单层！，OOM根源） |
| `--model_channels` | `32` | 直接影响 C | 32→209MB/层（temporal），16→105MB/层 |
| `--train_batch_size` | `1` | 直接乘以显存 | batch=1：~209MB；batch=4：~840MB（OOM） |
| `--channel_mult` | `1 2 4 4` | 决定有几个下采样级，每级都有时序Conv1d | 4级=4个时序卷积层，最大在 level=0（128×128） |

**当前配置（B=1，model_channels=32，num_temporal_latent=2）各层显存：**

| 分辨率 | 通道数 | 时序Conv1d中间激活 |
|--------|--------|------------------|
| 128×128 | 32 | 1×16384×2×32×100×4B = **419 MB** |
| 64×64 | 64 | 1×4096×2×64×100×4B = **210 MB** |
| 32×32 | 128 | 1×1024×2×128×100×4B = **105 MB** |
| 16×16 | 128 | 1×256×2×128×100×4B = **26 MB** |

反向传播需保留中间激活，总显存峰值约 **8–12 GB**（加上空间 UNet 本身的激活）。

#### 影响训练时间的参数 ← 主要瓶颈

**实测基准**（validate，B=1，accum=1，model_channels=16，num_temporal_latent=1）：
`3.79 optimizer steps/sec`（每次 forward/backward ≈ **0.26 sec**）

实际计算量 = `num_steps × accumulation_steps` 次 forward/backward。

| 参数 | 默认值 | 耗时影响 | 建议（baseline） |
|------|--------|---------|----------------|
| `--num_steps` | `25001` | 线性 | **10000–15000** |
| `--accumulation_steps` | `16` | **线性乘以 16**，是最大时间杀手 | **4**（保留梯度稳定性同时省 4×） |
| `--train_batch_size` | `1` | 增大会 OOM | 保持 **1** |
| `--save_model_iters` | `5000` | 每隔多少步保存一次 checkpoint（I/O 开销小可忽略） | **2000** |

**方案对比（FTM 固定 ~15 min）：**

| 方案 | num_steps | accumulation_steps | GPSD 耗时 | 总耗时 | warmup 建议 |
|------|-----------|--------------------|----------|--------|------------|
| 原始 | 25001 | 16 | ~29h | ~29h | 5000 |
| **推荐** | **15000** | **4** | **~4.4h** | **~4.7h** | **1500** |
| 激进 | 5000 | 1 | ~22min | ~40min | 500 |
| 折中 | 25000 | 1 | ~1.8h | ~2h | 2500 |

---

## 阶段三：Inference (`inference_RB.py`)

### 任务
对 12 条测试序列，用训练好的扩散模型做后验采样（MPDPS），再用 FTM 基函数解码到 HR 场。

---

### 参数表

#### 影响模型表现（输出质量）的参数

| 参数 | 默认值 | 含义 | 调大影响 | 调小影响 |
|------|--------|------|---------|---------|
| `--total_steps` | `20` | ODE 采样去噪步数。**这才是"推理步数"**，与训练无关 | 采样质量更好（收敛更好） | 采样质量略降，速度更快 |
| `--MPDPS` | `0.4` | 消息传递后验权重：Stage2（时序传播）相对 Stage1（直接观测）的强度 | 更依赖时序相关性 | 更依赖当帧直接观测 |
| `--zeta` | `0.009` | 后验梯度步长：每个 ODE step 加入多少观测约束 | 更强的观测约束，但过大会导致采样不稳定 | 约束弱，采样接近无条件生成 |
| `--sigma_min/max/rho` | 同训练 | ODE 步长调度，必须与训练一致 | 不要改 | — |

#### 必须与训练保持一致的参数（加载模型时）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_channels` | `32` | 必须与 train_GPSD_RB.py 中一致 |
| `--channel_mult` | `1 2 4 4` | 同上 |
| `--num_blocks` | `4` | 同上 |
| `--num_temporal_latent` | **8（需改为2）** | **当前默认值有误，必须与训练一致，否则模型加载失败** |
| `--layers_per_block` | `4` | 同上 |

#### 影响显存的参数

推理时不保留梯度，显存约为训练的 1/3，当前配置（B=1，model_channels=32，num_temporal_latent=2）推理显存约 **3–5 GB**，无问题。

#### 影响推理时间的参数

| 参数 | 默认值 | 耗时说明 |
|------|--------|---------|
| `--total_steps` | `20` | 每条序列约 9s，12条合计 **~2 min**。每步内含 MPDPS 内层循环（100次观测×矩阵乘），是主要开销 |

> 推理总体很快，无需优化。

---

## 已知 Bug

### `torch.kron` shape 错误
**影响文件**：`inference_RB.py` 第 143 行，`validate_pipeline.py` 已修复。

**原因**：某些 PyTorch 版本的 `torch.kron` 对高维张量返回展平结果。

**`inference_RB.py` 第 143 行需改为**：
```python
# 原来（有bug）：
temp[:, t_remove_group, :] = torch.kron(coeff.unsqueeze(2), post.unsqueeze(0).unsqueeze(0))

# 修复后：
temp[:, t_remove_group, :] = coeff.unsqueeze(-1) * post.float().view(1, 1, -1)
```

### `inference_RB.py` 的 `--num_temporal_latent` 默认值错误
当前默认值为 8，但训练时已改为 2。加载模型时会因网络结构不匹配而崩溃。
需将第 297 行默认值从 8 改为 2。

---

## 单卡显存实测（mc=32 全部 OOM，mc=16 结果如下）

> 测试条件：batch_size=1，T=100，单张 RTX 4090 24GB，含 optimizer step

| 配置 | 峰值显存 | 结论 |
|------|---------|------|
| mc=32, nb=4, ntl=2, fp32 | OOM (>24GB) | 不可用 |
| mc=32, nb=4, ntl=2, bf16 | OOM (24.18GB) | 不可用 |
| mc=16, nb=4, ntl=2, fp32 | **23.51GB** | ✅ 勉强可用（余量仅500MB） |
| mc=16, nb=4, ntl=4, fp32 | OOM (24.26GB) | 不可用 |

**关于4张4090能否增大 ntl 的结论**：
DDP（数据并行）是每张卡装完整模型 + 各处理1条序列，每张卡仍需23.5GB。4卡不能合并显存，**对提升 ntl 没有任何帮助**。

4卡的实际收益是：每个 optimizer step 4张卡各处理1条序列 → effective batch=4 → 梯度更稳定，不需要 accumulation_steps，训练速度 2-3×。

如果一定要 ntl=4，唯一可行方案是对 UNet 空间块启用 gradient checkpointing（反向传播时重新计算激活值），可以省约 2-3× 激活值显存，代价是训练速度降低 ~30%。

## 推荐的 4~5 小时 Baseline 配置

### 单卡方案（~5h）

```bash
# FTM：~15 min
CUDA_VISIBLE_DEVICES=0 conda run -n cvpr python3 train_FTM_RB.py \
    --max_iter 1500

# GPSD：~4.5h（mc=16 是单卡24GB唯一可用配置）
CUDA_VISIBLE_DEVICES=0 conda run -n cvpr python3 train_GPSD_RB.py \
    --core_path ./data/core_rb_1x128x128_<时间戳>.mat \
    --num_steps 15000 \
    --accumulation_steps 4 \
    --train_batch_size 1 \
    --warmup 1500 \
    --save_model_iters 2000 \
    --model_channels 16 \
    --num_temporal_latent 2

# Inference：~2 min
CUDA_VISIBLE_DEVICES=0 conda run -n cvpr python3 inference_RB.py \
    --basis_path ./ckp/basis_rb_<时间戳>.pth \
    --model_path ./exps/.../checkpoints/ema_14000.pth \
    --core_mean_std_path ./exps/.../core_mean_std.mat \
    --norm_stats_path ./data/norm_stats_rb_<时间戳>.json \
    --model_channels 16 \
    --num_temporal_latent 2 \
    --total_steps 20
```

### 4卡 DDP 方案（~2h，更快、梯度更稳定）

```bash
# GPSD：4卡各跑1条序列，effective batch=4，不需要accumulation
CUDA_VISIBLE_DEVICES=0,5,6,7 conda run -n cvpr torchrun --nproc_per_node=4 train_GPSD_RB.py \
    --core_path ./data/core_rb_1x128x128_<时间戳>.mat \
    --num_steps 15000 \
    --accumulation_steps 1 \
    --train_batch_size 1 \
    --warmup 1500 \
    --save_model_iters 2000 \
    --model_channels 16 \
    --num_temporal_latent 2
```
> 注意：`train_GPSD_RB.py` 目前不支持 DDP，需要添加 `torch.distributed` 初始化才能使用4卡方案。
