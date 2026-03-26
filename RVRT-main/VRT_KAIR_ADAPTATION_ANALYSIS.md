# VRT (KAIR) 模型分析与适配文档

## 一、原始模型 (KAIR)

### 1.1 输入格式
- **数据格式**: 视频序列，shape `(B, T, C, H, W)`
  - B: batch size
  - T: 时间帧数
  - C: 通道数（原始设计为 **3通道 RGB**）
  - H, W: 空间分辨率
- **原始设计假设**:
  - 输入为 RGB 视频（3通道）
  - LR 空间分辨率 ≥ 64×64（SPyNet 金字塔要求）
  - 上采样倍数 scale=4（默认）

### 1.2 模型组件与处理流程

#### 核心组件
1. **SPyNet（光流估计）**
   - 预训练于 RGB 数据，**硬编码 3通道输入**
   - 5级金字塔结构，每级下采样 2×，**要求输入 ≥ 64×64**
   - 输出多尺度光流场（4个尺度）

2. **Deformable Alignment（可变形对齐）**
   - 使用 SPyNet 光流对相邻帧进行 warp
   - `pa_frames=2`: 对齐 t-2, t-1, t+1, t+2 到当前帧 t
   - 使用 Deformable Convolution 进行精细对齐
   - 原始配置 `deformable_groups=16`

3. **Transformer Backbone**
   - 多阶段 3D 窗口注意力（时空联合建模）
   - `window_size=[6,8,8]`: 时间窗口6帧，空间窗口8×8
   - 下采样路径: scale=1 → 2 → 4 → 8
   - 上采样路径: scale=8 → 4 → 2 → 1
   - 使用 gradient checkpoint 节省显存

4. **重建模块**
   - 上采样到 HR 分辨率（scale=4）
   - 输出 3通道 RGB

### 1.3 输出格式
- **shape**: `(B, T, C, H_hr, W_hr)`
- **值域**: [0, 1]（归一化后的 RGB）

---

## 二、修改版本 (KAIR-master)

### 2.1 适配目标
将 VRT 从 RGB 视频超分辨率适配到 **单通道物理场数据**（RayleighBenard 数据集）：
- HR: 128×128，LR: 32×32，scale=4
- **单通道**（温度场）
- 100帧/序列，120条序列

### 2.2 关键修改

#### 修改 1: `models/select_network.py` (L210-211)
**原始代码**:
```python
netG = net(upscale=opt_net['upscale'],
           img_size=opt_net['img_size'],
           # 缺少 in_chans/out_chans 参数
```

**修改后**:
```python
netG = net(upscale=opt_net['upscale'],
           in_chans=opt_net.get('in_chans', 3),    # 新增
           out_chans=opt_net.get('out_chans', 3),  # 新增
           img_size=opt_net['img_size'],
```

**影响**: 允许通过配置文件指定输入/输出通道数，支持单通道数据。

---

#### 修改 2: `models/network_vrt.py` — SPyNet 单通道兼容 (L1496-1499)
**原始代码**:
```python
def get_flow_2frames(self, x):
    b, n, c, h, w = x.size()
    x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    # 直接传入 SPyNet，c=1 时报错
    flows_backward = self.spynet(x_1, x_2)
```

**修改后**:
```python
def get_flow_2frames(self, x):
    b, n, c, h, w = x.size()
    x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    # 单通道复制为3通道
    if c != 3:
        x_1 = x_1.repeat(1, 3, 1, 1)
        x_2 = x_2.repeat(1, 3, 1, 1)
```

**影响**: 单通道输入通过复制成3通道，绕过 SPyNet 的通道数限制。

---

#### 修改 3: `models/network_vrt.py` — SPyNet 小尺寸兼容 (L1501-1519)
**问题**: LR 32×32 经过 SPyNet 5级金字塔（每级÷2）后，第5级变为 1×1，初始 flow 计算 `floor(1/2)=0×0` → 报错。

**修改后**:
```python
# 临时上采样到 64×64
spy_h, spy_w = max(h, 64), max(w, 64)
if spy_h != h or spy_w != w:
    x_1 = F.interpolate(x_1, size=(spy_h, spy_w), mode='bilinear', align_corners=False)
    x_2 = F.interpolate(x_2, size=(spy_h, spy_w), mode='bilinear', align_corners=False)

def _rescale_flows(flows):
    """光流插值回原始尺寸并缩放位移值"""
    if spy_h == h and spy_w == w:
        return flows
    rescaled = []
    for i, flow in enumerate(flows):
        tgt_h, tgt_w = h // (2 ** i), w // (2 ** i)
        flow = F.interpolate(flow, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
        flow[:, 0] *= w / spy_w   # x方向位移缩放
        flow[:, 1] *= h / spy_h   # y方向位移缩放
        rescaled.append(flow)
    return rescaled
```

**影响**: 支持 32×32 等小尺寸输入，光流计算在 64×64 上进行，结果缩放回原始尺寸。

---

#### 修改 4: `options/vrt/010_train_vrt_videosr_rb.json` — 配置调整
**关键参数**:
```json
{
  "n_channels": 1,                    // 单通道
  "netG": {
    "in_chans": 1,
    "out_chans": 1,
    "img_size": [6, 32, 32],          // T=6, H=W=32
    "window_size": [6, 4, 4],         // 空间窗口改为4（原8会超出特征图）
    "deformable_groups": 8,           // 原16不整除embed_dim=64
    "use_checkpoint_attn": false,     // 关闭以避免DDP冲突
    "use_checkpoint_ffn": false
  },
  "find_unused_parameters": true      // DDP下冻结参数需要
}
```

**影响**:
- `window_size=[6,4,4]`: 避免深层 stage（scale=8）特征图 4×4 小于窗口 8×8
- `deformable_groups=8`: 满足 `embed_dim % deformable_groups == 0`
- `use_checkpoint_*=false`: 避免 DDP 下 segfault（或需加 `use_reentrant=False`）

---

#### 修改 5: `data/dataset_video_rb.py` — 新建数据集
**功能**:
- 加载 `.npz` 文件（`output` 字段，shape `(100, H, W)`）
- Min-max 归一化到 [0, 1]
- 训练时空间增强（翻转、转置）+ 时间镜像
- 返回格式: `{'L': lq, 'H': gt, 'key': ...}`（KAIR 标准格式）

**影响**: 提供与 KAIR 框架兼容的数据接口。

---

#### 修改 6: `main_test_vrt_rb.py` — 测试脚本
**功能**:
- 加载模型和 checkpoint
- 推理整个测试集
- 反归一化到物理值
- 保存 `pred.npz`, `gt.npz`, `lr.npz`, `meta.json`

**输出格式**:
- shape: `(N_frames, H, W, 1)`
- 值域: 物理值（已反归一化）

---

### 2.3 其他修复（见 ADAPTATION_NOTES.md）
1. **Padding bug**: `_test_clip` 空间 padding 公式修复（避免 einops 报错）
2. **DDP 冲突**: `checkpoint.checkpoint(..., use_reentrant=False)`
3. **冻结参数**: `find_unused_parameters=true`
4. **时间维度**: `num_frame_testing=0` 走整段推理路径

---

## 三、数据流总结

### 原始 KAIR
```
输入: (B, T, 3, 64, 64) RGB视频
  ↓
SPyNet: 3通道光流估计 → 多尺度光流
  ↓
Deformable Alignment: 帧对齐
  ↓
Transformer: 时空注意力 (window=[6,8,8])
  ↓
Upsample: scale=4
  ↓
输出: (B, T, 3, 256, 256) RGB视频
```

### 修改后 KAIR-master
```
输入: (B, T, 1, 32, 32) 单通道物理场
  ↓
SPyNet: 1→3通道复制 + 临时上采样到64×64 → 光流 → 缩放回32×32
  ↓
Deformable Alignment: 帧对齐 (deformable_groups=8)
  ↓
Transformer: 时空注意力 (window=[6,4,4])
  ↓
Upsample: scale=4
  ↓
输出: (B, T, 1, 128, 128) 单通道物理场
```

---

## 四、影响分析

### 4.1 架构层面
- **通道数**: 从 RGB 3通道 → 单通道，模型参数量略减
- **空间尺寸**: 从 64×64 → 32×32，特征图更小，计算量降低
- **窗口大小**: 空间窗口 8×8 → 4×4，感受野减小

### 4.2 性能影响
- **SPyNet 光流**: 单通道复制为3通道，光流质量可能略降（预训练权重针对 RGB）
- **小尺寸上采样**: 32×32 临时上采样到 64×64 再缩放回来，引入插值误差
- **窗口缩小**: 4×4 窗口在 32×32 输入上覆盖更大比例，但深层特征图（4×4）窗口覆盖全局

### 4.3 训练稳定性
- **DDP 兼容**: 需要 `find_unused_parameters=true` + `use_reentrant=False`
- **显存占用**: 小尺寸输入 + 单通道，显存需求大幅降低

---

## 五、文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `models/network_vrt.py` | 修改 | SPyNet 单通道兼容 + 小尺寸兼容 |
| `models/select_network.py` | 修改 | VRT 实例化透传 `in_chans`/`out_chans` |
| `data/dataset_video_rb.py` | 新建 | RB 数据集类 |
| `data/select_dataset.py` | 修改 | 注册新数据集 |
| `options/vrt/010_train_vrt_videosr_rb.json` | 新建 | 训练配置 |
| `main_test_vrt_rb.py` | 新建 | 测试脚本 |
| `ADAPTATION_NOTES.md` | 新建 | 踩坑记录 |

---

## 六、使用方法

### 训练
```bash
cd /data/yc/KAIR-master
CUDA_VISIBLE_DEVICES=0,5,6,7 torchrun --nproc_per_node=4 main_train_vrt.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json --dist true
```

### 测试
```bash
python main_test_vrt_rb.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json \
    --checkpoint experiments/010_train_vrt_videosr_rb/models/200000_G.pth \
    --output-dir ./output_rb
```

### 输出
- `pred.npz`: 预测结果，shape `(N, H, W, 1)`，物理值
- `gt.npz`: 真值，shape `(N, H, W, 1)`，物理值
- `lr.npz`: LR 输入，shape `(N, h, w, 1)`，物理值
- `meta.json`: 元数据（帧数、序列数、归一化统计、checkpoint 路径）
