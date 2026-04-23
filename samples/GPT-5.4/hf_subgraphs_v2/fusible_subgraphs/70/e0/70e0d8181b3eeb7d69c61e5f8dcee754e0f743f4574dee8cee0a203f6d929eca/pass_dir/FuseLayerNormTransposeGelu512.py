import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _layer_norm_gelu_transpose_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_time,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_start = pid * n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    y = x_centered * rstd
    y = y * w + b

    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    batch_idx = pid // n_time
    time_idx = pid % n_time
    out_offsets = batch_idx * (n_cols * n_time) + cols * n_time + time_idx
    tl.store(out_ptr + out_offsets, y, mask=mask)


@torch.fx.wrap
def fused_layer_norm_transpose_gelu_512(bias, weight, x):
    n_time = x.shape[-2]
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    n_batch = n_rows // n_time

    out = torch.empty((n_batch, n_cols, n_time), device=x.device, dtype=x.dtype)

    _layer_norm_gelu_transpose_kernel[(n_rows,)](
        x,
        weight,
        bias,
        out,
        n_rows,
        n_time,
        n_cols,
        1e-05,
        BLOCK_SIZE=512,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_layer_norm_transpose_gelu_512