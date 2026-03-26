exp_name = 'basicvsr_plusplus_c64n7_kf256_1ch_4x'

# ── Model ──────────────────────────────────────────────────────────────────────
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        in_channels=1,                          # single-channel physical field
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                          'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

train_cfg = dict(fix_iter=1000)   # freeze SPyNet for first 1000 iters
test_cfg  = dict(metrics=[], crop_border=0)

# ── Data paths ─────────────────────────────────────────────────────────────────
_hr_path = '/data/yc/dataset/KF256_data/x4/HR'
_lr_path = '/data/yc/dataset/KF256_data/x4/LR'

# Pipeline: dataset returns tensors directly, so only Collect is needed
train_pipeline = [
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
]
val_pipeline = [
    dict(type='Collect', keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key']),
]
test_pipeline = val_pipeline

# ── Datasets ───────────────────────────────────────────────────────────────────
data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    train=dict(
        type='SRKFDataset',
        hr_path=_hr_path,
        lr_path=_lr_path,
        pipeline=train_pipeline,
        scale=4,
        split='train',
        num_samples=70,
        num_input_frames=30,       # random 30-frame clip per sample
        train_ratio=0.8,
        valid_ratio=0.1,
        use_mirror_sequence=True,  # doubles clip length to 60 frames
        test_mode=False),

    val=dict(
        type='SRKFDataset',
        hr_path=_hr_path,
        lr_path=_lr_path,
        pipeline=val_pipeline,
        scale=4,
        split='valid',
        num_samples=70,
        num_input_frames=None,     # full sequence
        train_ratio=0.8,
        valid_ratio=0.1,
        use_mirror_sequence=False,
        test_mode=True),

    test=dict(
        type='SRKFDataset',
        hr_path=_hr_path,
        lr_path=_lr_path,
        pipeline=test_pipeline,
        scale=4,
        split='test',
        num_samples=70,
        num_input_frames=None,     # full sequence
        train_ratio=0.8,
        valid_ratio=0.1,
        use_mirror_sequence=False,
        test_mode=True),
)

# ── Optimizer ──────────────────────────────────────────────────────────────────
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# ── LR schedule ────────────────────────────────────────────────────────────────
total_iters = 10000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[10000],
    restart_weights=[1],
    min_lr=1e-7)

# ── Logging & checkpointing ────────────────────────────────────────────────────
checkpoint_config = dict(interval=5000, save_optimizer=False, by_epoch=False, max_keep_ckpts=2)
evaluation = dict(interval=5000, save_image=False)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# ── Runtime ────────────────────────────────────────────────────────────────────
find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level  = 'INFO'
work_dir   = f'./work_dirs/{exp_name}'
load_from  = None
resume_from = None
workflow   = [('train', 1)]
