import torch
import triton
import triton.language as tl


@triton.jit
def _spatial_mean_2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Reduce over (H, W) for one (batch, channel) slice.
    Grid: (N * C,).
    out shape: [N, C, 1, 1]  → stored flat as [N*C], offset = n*C + c.
    """
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = (n * C + c) * H * W
    total = H * W
    acc = 0.0

    for start in range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.sum(x, 0)

    mean_val = acc / total
    if DTYPE == 'float16':
        tl.store(out_ptr + n * C + c, mean_val.to(tl.float16))
    elif DTYPE == 'bfloat16':
        tl.store(out_ptr + n * C + c, mean_val.to(tl.bfloat16))
    else:
        tl.store(out_ptr + n * C + c, mean_val.to(tl.float32))


@torch.fx.wrap
def fast_spatial_mean_2d(x):
    """
    Fast Triton replacement for x.mean((2, 3), keepdim=True).
    x   : [N, C, H, W]
    out : [N, C, 1, 1]
    """
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    if x.is_cuda:
        DTYPE = str(x.dtype).replace('torch.', '')
        _spatial_mean_2d_kernel[(N * C,)](
            x, out,
            N, C, H, W,
            DTYPE=DTYPE,
            BLOCK=256,
        )
    return out