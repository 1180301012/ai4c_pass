import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def _fused_residual_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    residual_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_id * n_cols
    offsets = row_start + col_offsets

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0)
    summed = x + residual
    summed_f32 = summed.to(tl.float32)

    mean = tl.sum(summed_f32, axis=0) / n_cols
    centered = summed_f32 - mean
    var = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    out = centered * inv_std
    out = out * weight + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_residual_layernorm_768(in_0, in_1, in_2, in_3):
    n_cols = in_2.shape[-1]
    n_rows = in_2.numel() // n_cols
    out = torch.empty_like(in_2, dtype=torch.float32)

    block_size = 1024
    num_warps = 4 if n_rows < 256 else 8
    grid = (n_rows,)
    _fused_residual_layernorm_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        n_rows,
        n_cols,
        1e-07,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=4,
    )
    return out


def replacement_func():
    return fused_residual_layernorm_768