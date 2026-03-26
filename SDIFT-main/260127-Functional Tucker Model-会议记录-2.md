# 260127-Functional Tucker Model-会议记录

**会议信息：**

- **时间：** 2026-01-27 (按文件名 "260127" 记).
- **主题：** Functional Tucker Model (FTM) 的概念与实现细节梳理 (SDIFT 框架中的 FTM 部分).
- **参会：** 易闯，刘钲阳，陈朝野，会议用户 865404.
- **材料：** `会议文字转写.txt`，`Chen 等 - 2025 - Generating Full-field Evolution of Physical Dynamics from Irregular Sparse Observations.pdf`，`SDIFT/` (重点：`SDIFT/FTM_model.py`，`SDIFT/train_FTM.py`).

> **Note:** 这份记录优先复述会议里讲清楚的部分，并用论文与代码补齐术语. GPSD / MPDPS 等 diffusion 细节本次讨论几乎没展开，因此这里不展开.

---

## 1. FTM 我们到底在建模什么.

**目标：** 给定稀疏且 off-grid 的观测点 `(i, t, y)`，希望能在任意连续坐标 `(i, t)` 上预测场值 `y(i, t)`.

> **Note:** 对照 `SDIFT/` 的实现，FTM 主要负责两件事：1) 学到共享的 latent functions. 2) 把每个序列在离散时间步上的 dynamics 编码成一条 core sequence `W_{b,t}`，并在给定某个 `W_t` 时，用闭式 multilinear 运算推断 `y(i,t)` 的空间部分. 

**核心直觉：** 物理场是高维对象 (例如 512×512 的空间网格，再加时间). 直接在原空间学习很难. FTM 用一个 "低秩的 Tucker 结构" 把问题拆成两部分：

- **每个维度各自学一个函数：** 把该维度的坐标值映射到一个低维 latent vector (会议里也叫 latent factors / 隐空间表示).
- **用 core tensor 把这些 latent vectors 组合成输出：** core tensor 控制 "各维特征怎么交互"，从而得到某个点的物理量.

---

## 2. 从 Tucker 分解到 Functional Tucker.

**离散 Tucker (直观版)：**

- 数据张量 `Y ∈ R^{I_1×···×I_K}`.
- 第 `k` 维的因子矩阵 `U^{(k)} ∈ R^{I_k×R_k}`，`R_k` 是我们预先设定的压缩维度 (rank).
- 核心张量 `W ∈ R^{R_1×···×R_K}` 决定各维 latent factors 的交互模式.
- 某个位置的重构可以写成：

$$
\hat y(i_1,\ldots,i_K)
= \sum_{r_1,\ldots,r_K} W_{r_1,\ldots,r_K}\prod_{k=1}^{K}U^{(k)}_{i_k,r_k}.
$$

**Functional Tucker (会议里最重要的推广点)：**

- 现实观测经常不在固定网格点上 (off-grid)，而且我们希望任意分辨率都能取值.
- 因子矩阵 `U^{(k)}` 无法覆盖连续坐标，因此把它换成向量值函数 `f_k(i_k): R → R^{R_k}`.
- 对任意连续坐标 `i=(i_1,\ldots,i_K)`：

$$
\hat y(i)
= \operatorname{vec}(W)^{T}\bigl(f_1(i_1)\otimes\cdots\otimes f_K(i_K)\bigr).
$$

> **Note:** 会议里反复强调的 "因子矩阵 -> 函数"，就是为了让模型从 "只能在 on-grid 位置预测" 变成 "任意位置都能预测".

---

## 3. 两个概念随时间变化.

会议里的结论是：

- **latent functions 不随时间变：** `f_k(·)` 的参数在所有时间步共享 (shared).
- **随时间变的是 core tensor：** 对单个物理过程序列，可以写成 `W_t` (core sequence). 在训练代码里通常是对每个样本 `b` 都有一条 core 序列，因此更完整的记号是 `W_{b,t}`.
- **场值随时间变化的来源：** "每个维度的规律基本不变"，但 "规律之间怎么组合" 在随时间变.

用公式写就是：

$$
\hat y(i,t_m)
= \operatorname{vec}(W_{t_m})^{T}\bigl(f_1(i_1)\otimes\cdots\otimes f_K(i_K)\bigr).
$$

> **Note:** 会议里也做出了一个区分，当数据中有时间t这个维度时，把时间也当作一个维度，再加一个 time-net. 论文主线采用的是 "共享 latent functions + 不同时间 `W_t`" 这条路. 但从概念上说，time-net 和 `W_t` 的含义不同：time-net 更像在学 "时间这个特征本身" (时间步与时间步之间怎么相互影响)，而 `W_t` 是在学 "某个时间步内，各维特征怎么交互" 的系数.

---

## 4. 训练时到底学哪些东西.

**可学习参数：**

- **latent functions：** 每个维度一个 MLP，用来参数化 `f_k(·)` (会议里举例是 `U_net / V_net / W_net`).
- **core sequence：** 对单个序列是 "每个时间步一个 core tensor `W_t`". 在训练代码里通常是 "每个样本 `b`、每个时间步 `t` 一个 core tensor `W_{b,t}`".

**为什么大家会觉得它像 encoder-decoder：**

- 会议里出现的共识是：结构上更像 "encoder + 闭式重构". 坐标先被映射成 latent vectors，然后通过固定的 multilinear 运算直接得到 `ŷ`. 这里没有单独定义一个 decoder 网络去学映射.

**loss 是什么：**

- 主项是重构误差 (预测值 vs 真值). 论文里写的是平方误差项 (见公式 (8)). `SDIFT/train_FTM.py` 里用的是 RMSE (见 `loss_fn`).
- 额外加了一项时间平滑约束. 会议里口头说的是 "限制相邻时间步的物理量跳变". 论文/代码里的具体实现是对 core sequence 做 total variation (见论文公式 (8)，代码函数名是 `total_variation_loss`).

论文里的一个对应写法是：

$$
\mathcal{L}
= \mathbb{E}_{(i,t_m,y_{i,t_m})\sim D}\|y_{i,t_m}-\hat y(i,t_m)\|_2^2
+\beta\sum_{m=2}^{M}\|W_{t_m}-W_{t_{m-1}}\|_F^2.
$$

> **Note:** 会议讨论时没有严格区分 "对输出场做平滑" 还是 "对 core 序列做平滑". 这里按论文与代码的实现记录为 "对 core 序列做平滑".

> **Note:** `SDIFT/utils.py` 里的 `total_variation_loss` 是一个实现版本，写法上不完全等同于 "逐时间差的 Frobenius norm 求和" 这种教科书定义，但目标是一致的：惩罚 core 在时间维度上的剧烈变化.

---

## 5. Rank (压缩维度) 怎么选.

会议里明确提到：

- `R_k` 是 preset 的超参数，我们要决定每个维度压缩到什么程度.

> **Note:** 下面两条是整理时的补充 (不是会议原话).
>
> - `R_k` 越小，未知数更少，通常更容易拟合稀疏观测，但表达能力更弱.
> - `R_k` 越大，表达能力更强，但优化更难，计算与内存也更贵.

---

## 6. 会上对应用场景的判断.

- **关于 "稀疏观测能不能还原"：** 会议里认为 paper 的 claim 是 "非常稀疏也能还原"，但很可能会 "往大了吹". 结论是可以试，但要用自己的数据做验证.
- **关于 "没有真值怎么训练"：** 至少要有观测点 `(i,t,y)` 的真值来算 loss. 如果完全没有 `y`，监督训练做不了.

> **Note:** 补充一句直觉 (不是会议原话)：FTM 的 low-rank 结构把未知数更多压在 core 上 (相比直接拟合全场)，因此在稀疏观测场景里通常更有希望.

---

## Q&A (基于讨论整理)

**Q1：FTM 可以理解成 encoder-decoder 吗.**

**A：** 更像 encoder + 闭式重构. MLP 学的是各维 latent function，core tensor 学的是交互系数. 输出 `ŷ` 是为了和真值算 loss，但没有单独训练一个 decoder 网络.

**Q2：时间序列信息具体体现在哪.**

**A：** `f_k(·)` 不随时间变，时间用来选取 `W_t`. 另外会加一个相邻时间步的连续性约束 (实现上常对 core 序列做 TV)，避免时序上出现太大跳变.

**Q3：什么叫 "共享 (shared)".**

**A：** 指各维的 MLP 参数在所有时间步共享. 不共享的是 core tensor：每个时间步 `t` 有自己的 `W_t`.

**Q4：core tensor 怎么把多个维度的特征融合起来.**

**A：** core 的维度由各个 `R_k` 决定. 计算上就是把每个维度的 latent vector 做 Kronecker / 外积，再和 `vec(W_t)` 做内积 (等价于 multilinear product).

**Q5：loss 是什么，为什么要加第二项.**

**A：** 主项是 MSE (预测值 vs 真值). 第二项是时间连续性约束：会议里口头解释为 "限制相邻时间步的物理量跳变". 实现上常用对 core sequence 的平滑正则 (例如 TV)，直觉是 "真实物理过程通常是连续的"，不希望相邻时间步突然大跳.

**Q6：如果目标数据没有完整真值，还能不能用.**

**A：** 至少要有稀疏观测点 `(i,t,y)` 作为监督，才能训练 FTM. 如果完全没有 `y` 的真值，监督式训练走不通.

**Q7：输入会把 time t 直接喂进各个 MLP 吗.**

**A：** 会议里提到当前实现不是：每个 MLP 只吃一个维度的坐标值. 时间变化通过 `W_t` 体现. 当然有的数据还有时间t：额外加 time-net，把时间当作一个维度建模

---

## 7. SDIFT 代码对照 (关键片段 + 解释)

**符号对照 (论文/会议 -> 代码)：**

- `f_k(·)` -> `Tensor_inr_3D` 里的 `U_net / V_net / W_net` (每个维度一个 MLP).
- `W_{b,t}` -> `train_FTM.py` 里的 `tucker_core[b, t, ...]` (可学习 core sequence).
- 观测点 `(i,t,y)`
- `y(i,t)` 的闭式重构 -> `einsum` 链式乘法 (训练与推断都一样).

**核心张量形状对照 (先把维度看清楚)：**

- `data` (训练数据)：`B×T×I1×I2×I3`，见 `SDIFT/train_FTM.py` 里 `data_size = data.size()`.
- `mask_tr` (训练观测 mask)：`T×I1×I2×I3`，在训练时会复制成 `B×T×I1×I2×I3` 用于只在观测点上算 loss.
- `tucker_core` (可学习 core sequence)：`B×T×R1×R2×R3`.
- `basises = (U^{(1)}, U^{(2)}, U^{(3)})`：分别是 `I1×R1`，`I2×R2`，`I3×R3`.
- `output` (重构全场)：`B×T×I1×I2×I3`.

> **Note:** SDIFT 的训练实现是 "先重构全场，再用 mask 选出观测点算 loss". 这和论文的 "只对观测点求期望" 在数学目标上等价，但在实现上更直接，也更费算力.

**坐标与稀疏观测 mask 是怎么来的：**

下面这段代码来自 `SDIFT/preprocessing_data.py`，它把每个维度的离散索引归一化到 `[0,1]`，再随机生成一个稀疏观测 mask (`mask_tr`) 作为训练监督子集.

```python
# SDIFT/preprocessing_data.py
t_ind_uni = np.linspace(0, self.shape[1]-1, self.shape[1]).astype(int)/(self.shape[1]-1)
if self.shape[2] == 1:
    u_ind_uni = np.ones_like([1])
else:
    u_ind_uni = np.linspace(0, self.shape[2]-1, self.shape[2]).astype(int)/(self.shape[2]-1)
v_ind_uni = np.linspace(0, self.shape[3]-1, self.shape[3]).astype(int)/(self.shape[3]-1)
w_ind_uni = np.linspace(0, self.shape[4]-1, self.shape[4]).astype(int)/(self.shape[4]-1)

mask_tr = self.create_mask(self.shape[1:], r=0.1)  # shape: (T, I1, I2, I3)
```

> **Note:** 代码里观测点来自 on-grid 的随机子集，但坐标本身是 float，因此 `Tensor_inr_3D` 在接口层面是可以吃 off-grid 的连续坐标的 (见后面的 `mode="sampling"`).

**训练 mask 在代码里是怎么用的：**

下面这段代码来自 `SDIFT/train_FTM.py`，它把 `mask_tr (T×I1×I2×I3)` 扩展到 batch 维度，保证 loss 只在 `mask==1` 的位置计算.

```python
# SDIFT/train_FTM.py
mask_tmp = mask_tr.unsqueeze(0)
mask_tmp = mask_tr.repeat(data.shape[0], 1, 1, 1, 1)  # B×T×I1×I2×I3
...
loss = loss_fn(output, data, mask=mask_tmp)
```

> **Note:** 这里的 `mask_tr` 是对整个训练集共享的一张 mask (来自 metadata). 也就是说，所有样本 `b` 在同一组空间-时间位置有观测. 这和论文 problem statement 里 "每个时间步的观测模式可能变化" 不冲突，但比论文的设置更特殊 (更简单).

**latent functions 在代码里怎么参数化：**

下面这段代码来自 `SDIFT/FTM_model.py`，核心点是 "每个维度一个 MLP"，输入是 1 维坐标，输出是 `R_k` 维 latent vector. 其中 `SineLayer` 对应会议里提到的 "更擅长拟合周期性现象" 的激活设计.

```python
# SDIFT/FTM_model.py
class Tensor_inr_3D(nn.Module):
    def __init__(self, R: tuple, omega=10):
        ...
        self.U_net = nn.Sequential(
            SineLayer(1, mid_channel, omega_0=omega),
            SineLayer(mid_channel, mid_channel, omega_0=omega),
            nn.Dropout(0),
            nn.Linear(mid_channel, self.r_1),
            nn.Tanh(),
        )
        self.V_net = ...
        self.W_net = ...
```

**SineLayer 到底在做什么：**

下面这段代码来自 `SDIFT/FTM_model.py`. 它用一个线性层再套 `sin(sin(·))`. 直觉上这是在引入强周期性基，从而更容易拟合高频或周期结构 (会议里也提到这一点).

```python
# SDIFT/FTM_model.py
class SineLayer(nn.Module):
    def forward(self, input):
        return torch.sin(torch.sin(self.omega_0 * self.linear(input)))
```

> **Note:** 这里是双层 `sin`，不是常见的 SIREN 里的单层 `sin`. 这不影响我们把它理解为一种 "周期基函数参数化" 的实现

**`Continuous_Tucker_ssf` 这个类和主线有什么关系：**

`SDIFT/FTM_model.py` 里还有一个 `Continuous_Tucker_ssf`. 它直接实现了 "坐标 -> (U,V,W) -> Kronecker -> 和 core 向量内积 -> 输出标量" 这一条链路：

```python
# SDIFT/FTM_model.py
UV = self.kronecker_product_einsum_batched(U, V)
UVW = self.kronecker_product_einsum_batched(UV, W).squeeze(1)
out_put = torch.einsum("bi, i->b", UVW, self.core)
```

> **Note:** 它更像是公式层面的一个最小实现 (每次只输出某些点的值). 但当前主线训练脚本 `SDIFT/train_FTM.py` 用的是 `Tensor_inr_3D + einsum` 的写法，用来一次性重构整张网格场，再用 mask 选观测点算 loss.

**同一套网络为什么有 "training / sampling" 两种 mode：**

下面这段代码来自 `SDIFT/FTM_model.py`，它解释了会议里 "网络参数共享" 在实现上的具体样子：

- `mode="training"`：输入是每个维度的坐标列表 `u_ind_uni / v_ind_uni / w_ind_uni`，输出是 3 个因子矩阵 (basis matrices)，形状分别是 `I1×R1`，`I2×R2`，`I3×R3`.
- `mode="sampling"`：输入是任意的连续坐标点集 `n×3`，输出是每个点的 Kronecker 特征 `n×(R1R2R3)`，用于把 `vec(W_t)` 映射到观测值 `y`.

```python
# SDIFT/FTM_model.py
def forward(self, input_ind_train=None, input_ind_sampl=None):
    if self._mode == "training":
        U = self.U_net(input_ind_train[0].unsqueeze(1))  # I1 × R1
        V = self.V_net(input_ind_train[1].unsqueeze(1))  # I2 × R2
        W = self.W_net(input_ind_train[2].unsqueeze(1))  # I3 × R3
        return (U, V, W)
    elif self._mode == "sampling":
        U = self.U_net(input_ind_sampl[:, :1]).unsqueeze(1)   # n × 1 × R1
        V = self.V_net(input_ind_sampl[:, 1:2]).unsqueeze(1)  # n × 1 × R2
        W = self.W_net(input_ind_sampl[:, 2:3]).unsqueeze(1)  # n × 1 × R3
        UV = self.kronecker_product_einsum_batched(U, V)
        UVW = self.kronecker_product_einsum_batched(UV, W).squeeze(1)  # n × (R1R2R3)
        return UVW
```

**`kronecker_product_einsum_batched` 的输出到底是什么：**

下面这段代码来自 `SDIFT/FTM_model.py`. 它做的是 batched Kronecker product，把 `n×1×R1` 和 `n×1×R2` 组合成 `n×(1·1)×(R1·R2)`，再和第三个维度继续组合，最终得到 `n×(R1R2R3)` 的特征行向量.

```python
# SDIFT/FTM_model.py
def kronecker_product_einsum_batched(self, A, B):
    res = torch.einsum("bac,bkp->bakcp", A, B).view(
        A.size(0),
        A.size(1) * B.size(1),
        A.size(2) * B.size(2),
    )
    return res
```

> **Note:** 在推断阶段，我们可以把 `A = basis_function(input_ind_sampl=ind_conti)` 理解成论文里的观测算子矩阵 `A(·)`：每一行就是 `f_1(i_1) ⊗ f_2(i_2) ⊗ f_3(i_3)`.

**如果我们真的想把 time 当作一个维度，代码里有没有现成模块：**

有. `SDIFT/FTM_model.py` 里提供了 `Tensor_inr_4D`，它额外定义了一个 `T_net`，输入时间坐标并输出时间维的 latent vector. 这更贴近 "把 time 也当作一个 mode" 的设计.

```python
# SDIFT/FTM_model.py
class Tensor_inr_4D(nn.Module):
    def __init__(self, R: tuple, omega=10):
        ...
        self.T_net = nn.Sequential(..., nn.Linear(mid_channel, self.r_1), nn.Tanh())
        self.U_net = ...
        self.V_net = ...
        self.W_net = ...
```

> **Note:** 当前主线训练脚本 `SDIFT/train_FTM.py` 用的是 `Tensor_inr_3D`，也就是 "time 不进入 latent functions，time 通过 core sequence 承担" 这一版. `Tensor_inr_4D` 更像是会议里提到的 time-net 方案的雏形.

**core sequence `W_{b,t}` 在训练时如何参与重构：**

下面这段代码来自 `SDIFT/train_FTM.py`，它展示了训练时的关键张量形状与那条 `einsum` 链.

```python
# SDIFT/train_FTM.py
tucker_core = (torch.ones(data_size[0], data_size[1], R[0], R[1], R[2]) / 2).to(device)
tucker_core.requires_grad = True  # learnable core sequence W_{b,t}

basises = basis_function(input_ind_train=ind_input)  # (I1×R1, I2×R2, I3×R3)

output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind])
output = torch.einsum("nj, btmjk->btmnk", basises[1], output)
output = torch.einsum("ok, btmnk->btmno", basises[2], output)
```

> **Note:** 如果我们把 `tucker_core[b, t]` 看成 `W_{b,t} ∈ R^{R1×R2×R3}`，那上面的 3 步就是 `W_{b,t} ×_1 U ×_2 V ×_3 W` 的代码版本，最终得到 `ŷ[b,t] ∈ R^{I1×I2×I3}`.

**从 `einsum` 回到 Tucker 求和式：**

如果我把上面的 3 个 `einsum` 写成更贴近数学的形式，那么对任意一个 batch `b`，时间 `t`，空间网格点 `(m,n,o)`，输出是：

$$
\hat y_{b,t,m,n,o}
= \sum_{i=1}^{R1}\sum_{j=1}^{R2}\sum_{k=1}^{R3}
U^{(1)}_{m,i}\,U^{(2)}_{n,j}\,U^{(3)}_{o,k}\,\bigl(W_{b,t}\bigr)_{i,j,k}.
$$

这就是我们熟悉的 Tucker multilinear product，只是代码用 `einsum` 做了高效实现.

**把 `einsum` 字母对上维度会更直观：**

我把 `SDIFT/train_FTM.py` 的第一步 `einsum` 展开解释一下：

- `basises[0]` 的 `mi`：`m=I1`，`i=R1`.
- `tucker_core` 的 `btijk`：`b=B`，`t=T`，`i=R1`，`j=R2`，`k=R3`.
- 输出 `btmjk`：相当于把 `R1` 这个 mode 用 `U` 投影到 `I1`，剩下 `(R2,R3)` 还保留在 core 里.

对应代码如下：

```python
# SDIFT/train_FTM.py
output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind])
```

后两步完全类比：分别用 `V (I2×R2)`，`W (I3×R3)` 把 `R2`，`R3` 也投影回空间网格.

**稀疏监督是怎么落到 loss 上的：**

下面这段代码来自 `SDIFT/train_FTM.py`，它对应会议里 "只在观测点上算 loss"，并且把 TV 正则加在 core 序列上做平滑.

```python
# SDIFT/train_FTM.py
loss = loss_fn(output, data, mask=mask_tmp) \
  + total_variation_loss(tucker_core[batch_ind], weight=1e-7)
```

**FTM 在代码里是怎么被 "联合训练" 的：**

论文里提到 alternating-direction 的训练策略，但 `SDIFT/train_FTM.py` 的实现是更直接的 joint optimization：把 latent functions 的网络参数和 `tucker_core` 一起丢进同一个 AdamW.

```python
# SDIFT/train_FTM.py
params = []
params += [x for x in basis_function.parameters()]
tucker_core.requires_grad = True
params += [tucker_core]
optimizer = optim.AdamW(params, learning_rate)
```

> **Note:** 这两种做法的目标一致，都是在拟合 latent functions 和 core. 只是优化路径不同，可能会影响收敛速度与稳定性.

**第一阶段产物怎么被保存给第二阶段用：**

下面这段代码来自 `SDIFT/train_FTM.py`. 当评估 RMSE 创新低时，它会把 `tucker_core` (所有训练样本的 core sequence) 存成 `.mat`，并把 `basis_function` (包含 3 个 MLP 的模块) 存成 `.pth`.

```python
# SDIFT/train_FTM.py
scio.savemat("./data/core_" + config.data_name + "... .mat",
             {"core": tucker_core.detach().cpu().numpy()})
torch.save(basis_function, "./ckp/basis_" + config.data_name + "... .pth")
```

> **Note:** 这也解释了为什么 `SDIFT/message_passing_DPS.py` 里会直接 `torch.load(basis_path)` 去加载 basis_function 做解码.

**`total_variation_loss` 实现细节 (它惩罚的到底是什么)：**

下面这段代码来自 `SDIFT/utils.py`. 关键点是 `torch.norm(..., dim=(1))` 其实是在时间差分维度 (T-1) 上做一个 L2 聚合，再把所有 core 元素加起来.

```python
# SDIFT/utils.py
diff = X[:, 1:, :, :, :] - X[:, :-1, :, :, :]
tv_loss = torch.sum(torch.norm(diff, p='fro', dim=(1)))
```

> **Note:** 这更接近一种 "对每个 core 元素沿时间的 L2 变化量做惩罚" 的正则，而不是严格的 `\sum_t \|W_{t}-W_{t-1}\|_F`. 但从建模目的上，它仍然在鼓励时间维度上的平滑.