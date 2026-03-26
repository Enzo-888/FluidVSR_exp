# VRT (KAIR) 适配 RayleighBenard 数据集 — 踩坑记录

## 改动文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `models/network_vrt.py` | 修改 | SPyNet 单通道兼容 + 小尺寸输入兼容 + checkpoint fix |
| `models/select_network.py` | 修改 | VRT 实例化时透传 `in_chans`/`out_chans` |
| `data/dataset_video_rb.py` | 新建 | RB 数据集类 |
| `data/select_dataset.py` | 修改 | 注册新数据集 |
| `options/vrt/010_train_vrt_videosr_rb.json` | 新建 | 训练配置 |
| `main_test_vrt_rb.py` | 新建 | 测试脚本 |

---

## 主要难点

### 1. SPyNet 不支持单通道输入
**问题**：SPyNet 在 RGB（3通道）数据上预训练，`get_flow_2frames()` 直接把 LR 帧传入，1通道会报维度错误。
**修复**：传入前 `repeat(1, 3, 1, 1)` 复制成3通道。

### 2. SPyNet 不支持小于 64×64 的输入
**问题**：SPyNet 有 5 级金字塔，每级 avg_pool2d÷2。输入 32×32 时，第5级后变为 1×1，初始 flow = `floor(1/2)=0×0` → 报错。
**修复**：`get_flow_2frames()` 中临时上采样到 64×64，光流算完后插值回原始尺寸并缩放位移值（×32/64=0.5）。

### 3. window_size 大于深层 stage 的特征图尺寸
**问题**：VRT 有 scale=8 的下采样阶段，LR 32×32 经过 8× 后变为 4×4，但 `window_size=[6,8,8]`，窗口比特征图还大，导致 position_bias 形状不匹配。
**修复**：`window_size` 改为 `[6,4,4]`（4 ≤ 32//8=4 ✓）。

### 4. deformable_groups 不整除 embed_dim
**问题**：新版 torchvision 要求 `input_channels % deformable_groups == 0`，原始配置 `embed_dim=120, deformable_groups=16`，120%16=8≠0。
**修复**：`deformable_groups` 改为 8（120%8=0 ✓）。

### 5. gradient checkpoint 与 DDP 冲突导致 segfault
**问题**：`torch.utils.checkpoint` 默认 `use_reentrant=True`，在 DDP 下通过 hooks 追踪梯度，与 DDP 的 bucket 通信机制冲突 → segmentation fault。
**修复**：所有 `checkpoint.checkpoint()` 调用加 `use_reentrant=False`。

### 6. DDP 下冻结参数导致 bucket 报错
**问题**：训练前 5000 iter 冻结 `spynet`/`deform` 参数，这些参数无梯度，DDP 认为 reduction 未完成。
**修复**：配置中 `find_unused_parameters: true`。

### 7. val.num_frame_testing 导致时间维度不整除
**问题**：`num_frame_testing=100` 时，clip 为 100 帧，`_test_clip` 只做空间 padding，不做时间 padding。100 % window_size[0]=6 ≠ 0 → einops Rearrange 报错。
**修复**：`num_frame_testing=0`，走整段序列推理路径，代码自动对时间维度 padding 到 window_size[0] 的倍数。

### 8. _test_clip 空间 padding 公式有 bug → einops "axis of length 9"
**问题**：原始 padding 公式 `(h // ws + 1) * ws - h` 即使 h 已经整除 ws 也会多加一个 ws。
例：h=32, ws=4 → h_pad=4，填充后 36×36。经过 3 次 'down' stage（各÷2）：36→18→9，第 3 次 9÷2 失败 → einops `EinopsError: axis of length 9 in chunks of 2`。
另外当 pad=0 时，`lq[:, :, :, -0:, :]` 即 `lq[0:]`（全 tensor），cat 后空间翻倍，同样出错。
**修复**：改为 `(ws - h % ws) % ws`（整除时得 0），并加 `if pad:` guard。
同样对 `_test_video` 的时间 padding 做同等修复（`models/model_vrt.py` 第 150-152 行）。

---

## 训练命令

```bash
cd /data/yc/KAIR-master
CUDA_VISIBLE_DEVICES=0,5,6,7 torchrun --nproc_per_node=4 main_train_vrt.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json \
    --dist true
```

## 测试命令

```bash
python main_test_vrt_rb.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json \
    --checkpoint experiments/010_train_vrt_videosr_rb/models/100000_G.pth
```
