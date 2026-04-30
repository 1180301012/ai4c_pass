import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Minimal overhead layer norm kernel with correct masking."""
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input row and convert to fp32
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean (masked elements are 0, so sum is correct)
    mean = tl.sum(x, axis=0) * (1.0 / N)

    # Compute variance - zero out masked elements to avoid bias
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) * (1.0 / N)

    # Normalize
    rstd = tl.rsqrt(var + 1e-5)
    x_norm = x_centered * rstd

    # Apply affine transform
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    # Store
    tl.store(out_ptr + row_start + offsets, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_layer_norm(bias, weight, x):
    N = weight.shape[0]
    num_rows = x.numel() // N

    out = torch.empty_like(x)

    # Use smallest power-of-2 >= N for BLOCK_SIZE
    if N <= 16:
        BLOCK_SIZE = 16
        num_warps = 1
    elif N <= 32:
        BLOCK_SIZE = 32
        num_warps = 1
    elif N <= 64:
        BLOCK_SIZE = 64
        num_warps = 1
    elif N <= 128:
        BLOCK_SIZE = 128
        num_warps = 1
    elif N <= 256:
        BLOCK_SIZE = 256
        num_warps = 1
    elif N <= 512:
        BLOCK_SIZE = 512
        num_warps = 2
    elif N <= 1024:
        BLOCK_SIZE = 1024
        num_warps = 4
    else:
        BLOCK_SIZE = 2048
        num_warps = 4

    _layer_norm_fwd[(num_rows,)](
        x, weight, bias, out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )

    return out