import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_1):
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13


def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)


@triton.jit
def fused_rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offset = row_idx * stride_row

    # Load input row (bfloat16)
    x = tl.load(input_ptr + row_offset + col_offsets)

    # Convert to float32 for normalization
    x_f32 = x.to(tl.float32)

    # RMS computation: mean of squares
    x_sq = x_f32 * x_f32
    sum_val = tl.sum(x_sq, axis=0)
    rsqrt_val = tl.rsqrt(sum_val / N + 1e-06)

    # Normalize and apply weight
    x_norm = x_f32 * rsqrt_val
    weight = tl.load(weight_ptr + col_offsets)
    w = 1.0 + weight.to(tl.float32)
    result = x_norm * w

    # Store as bfloat16
    tl.store(out_ptr + row_offset + col_offsets, result.to(tl.bfloat16))


@torch.fx.wrap
def fused_rmsnorm(tmp_2, in_1):
    shape = tmp_2.shape
    N = shape[-1]
    num_rows = tmp_2.numel() // N

    out = torch.empty_like(tmp_2)

    fused_rmsnorm_kernel[(num_rows,)](
        tmp_2,
        in_1,
        out,
        N,
        N,
        2048,
    )

    return out


def replacement_func():
    return fused_rmsnorm