import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = row_id * n_cols + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z_f32 = (x + y).to(tl.float32)

    mean = tl.sum(tl.where(mask, z_f32, 0.0), axis=0) / n_cols
    diff = z_f32 - mean
    var = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd * weight + bias).to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layernorm_dispatch(in_0, in_1, in_2, in_3):
    n_cols = in_2.shape[-1]
    n_rows = in_2.numel() // n_cols
    out = torch.empty_like(in_2)

    if n_cols <= 16:
        block_size = 16
        num_warps = 1
    elif n_cols <= 768:
        block_size = 1024
        num_warps = 4
    else:
        block_size = 1024
        num_warps = 8

    _fused_add_layernorm_kernel[(n_rows,)](
        in_2,
        in_3,
        in_1,
        in_0,
        out,
        n_rows,
        n_cols,
        1e-05,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out