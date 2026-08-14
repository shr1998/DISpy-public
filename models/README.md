# Model Weights

Place a compatible automatic-picking checkpoint here as `unet1.pth`, or select
a `.pth`/`.pt` file from the picking window.

The checkpoint must be a PyTorch `state_dict` compatible with `UNet(6)` from
`unet.py`. Model binaries are intentionally excluded from Git because the
tested checkpoint is approximately 151 MB and exceeds GitHub's normal 100 MB
file limit.
