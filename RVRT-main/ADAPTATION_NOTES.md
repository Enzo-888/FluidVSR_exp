# RVRT 适配 RayleighBenard 数据集 — 踩坑记录

## 改动文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `models/network_rvrt.py` | 修改 | 单通道支持 + SpyNet小尺寸修复 + in_chans参数 |
| `data/dataset_video_rb.py` | 新建 | RB数据集类 |
| `main_train_rvrt_rb.py` | 新建 | 训练脚本 |
| `main_test_rvrt_rb.py` | 新建 | 测试脚本 |

---

## 主要改动

### 1. SpyNet 不支持单通道输入
**修复**：`compute_flow()` 中对 c≠3 的输入 `repeat(1, 3, 1, 1)` 复制成3通道。

### 2. SpyNet 不支持小于 64×64 的输入
**问题**：SpyNet 有6级金字塔，32×32 经5次 avg_pool2d÷2 后为1×1，初始 flow = floor(1/2)=0×0 → 报错。
**修复**：`compute_flow()` 中临时上采样到 64×64，flow 算完后插值回原始尺寸并缩放位移值（×32/64=0.5）。

### 3. 输出通道硬编码为3
**修复**：`RVRT.__init__` 新增 `in_chans` 参数，`feat_extract` 和 `conv_last` 均改用此参数。

---

## 模型配置（baseline精简版）

```python
RVRT(
    upscale=4, clip_size=2,
    img_size=[2, 32, 32], window_size=[2, 8, 8],
    num_blocks=[1, 1, 1], depths=[2, 2, 2],
    embed_dims=[64, 64, 64], num_heads=[4, 4, 4],
    deformable_groups=4, attention_heads=4,
    in_chans=1,
)
```
约束验证：
- embed_dims % num_heads: 64 % 4 = 0 ✓
- embed_dims % deformable_groups: 64 % 4 = 0 ✓
- window_size[1] ≤ H: 8 ≤ 32 ✓，且 32 % 8 = 0 ✓
- num_frame % clip_size: 6 % 2 = 0 ✓

---

## 训练命令

```bash
# 4 GPU DDP
cd /data/yc/RVRT-main
conda activate cvpr
CUDA_VISIBLE_DEVICES=0,5,6,7 torchrun --nproc_per_node=4 main_train_rvrt_rb.py

# 单 GPU
python main_train_rvrt_rb.py
```

## 测试命令

```bash
python main_test_rvrt_rb.py \
    --checkpoint experiments/rvrt_rb/20000_G.pth
```

输出：`experiments/rvrt_rb/test_predictions/pred.npz` / `gt.npz` / `lr.npz` / `meta.json`

---

## 注意事项

- **deform_attn**：首次 forward 时 JIT 编译 CUDA 扩展（`models/op/`），需要 ninja 和 CUDA 编译工具链。编译约需1~2分钟，属正常现象。
- **SpyNet 权重**：复用 `/data/yc/KAIR-master/model_zoo/vrt/spynet_sintel_final-3d2a1287.pth`，无需重新下载。
- **norm_stats**：首次训练时由 train split 生成 `RB_120_norm_stats.json`，测试脚本会自动读取。
