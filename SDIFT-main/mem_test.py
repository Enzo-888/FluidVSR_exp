import torch
from networks_edm import Spatial_temporal_UNet

device = torch.device('cuda:0')
_ = torch.zeros(1, device=device)

configs = [
    # (label, model_channels, num_blocks, ntl, use_bf16)
    ('mc=16 nb=4 ntl=2 fp32', 16, 4, 2, False),
    ('mc=16 nb=4 ntl=4 fp32', 16, 4, 4, False),
    ('mc=16 nb=4 ntl=8 fp32', 16, 4, 8, False),
    ('mc=32 nb=4 ntl=2 bf16', 32, 4, 2, True),
    ('mc=32 nb=4 ntl=4 bf16', 32, 4, 4, True),
    ('mc=32 nb=4 ntl=8 bf16', 32, 4, 8, True),
]

for name, mc, nb, ntl, use_bf16 in configs:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        unet = Spatial_temporal_UNet(
            in_channels=1, out_channels=1,
            num_blocks=nb, num_temporal_latent=ntl,
            attn_resolutions=[], model_channels=mc,
            channel_mult=[1,2,4,4], dropout=0,
            img_resolution=128, label_dim=0,
            embedding_type='positional', encoder_type='standard',
            decoder_type='standard', augment_dim=9,
            channel_mult_noise=1, resample_filter=[1,1],
        ).to(device).train()
        opt = torch.optim.Adam(unet.parameters(), lr=1e-4)
        x  = torch.randn(1, 100, 1, 128, 128, device=device)
        nl = torch.randn(1, 100, 1, device=device)
        tl = torch.linspace(0, 1, 100).view(1, -1, 1).to(device)

        if use_bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = unet(x, nl, tl)
            out.mean().backward()
        else:
            out = unet(x, nl, tl)
            out.mean().backward()

        opt.step()
        peak    = torch.cuda.max_memory_allocated(device) / 1e9
        nparams = sum(p.numel() for p in unet.parameters()) / 1e6
        print(f'OK  {name}  peak={peak:.2f}GB  params={nparams:.1f}M')
        del unet, opt, x, nl, tl, out
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f'OOM {name}  (reached {peak:.2f}GB)')
        torch.cuda.empty_cache()
