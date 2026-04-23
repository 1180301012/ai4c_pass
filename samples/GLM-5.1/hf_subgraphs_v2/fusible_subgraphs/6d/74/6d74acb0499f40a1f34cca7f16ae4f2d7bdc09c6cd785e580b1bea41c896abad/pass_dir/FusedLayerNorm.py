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
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    n_rows, n_cols,
    eps,
    stride_in2_row, stride_in2_col,
    stride_in3_row, stride_in3_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs and cast to float32
    in_2 = tl.load(in_2_ptr + row_idx * stride_in2_row + col_offsets * stride_in2_col, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_idx * stride_in3_row + col_offsets * stride_in3_col, mask=mask, other=0.0).to(tl.float32)
    x = in_2 + in_3

    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols

    # Compute centered x and variance
    x_centered = (x - mean) * mask
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Compute reciprocal sqrt
    rstd = tl.math.rsqrt(var + eps)

    # Normalize
    x_norm = x_centered * rstd

    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    out = w * x_norm + b

    # Store as float32
    tl.store(out_ptr + row_idx * stride_out_row + col_offsets * stride_out_col, out, mask=mask)


@torch.fx.wrap
def fused_layernorm(bias, weight, in_2, in_3):
    out = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)

    n_rows = in_2.shape[0] * in_2.shape[1]
    n_cols = in_2.shape[2]

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    fused_layernorm_kernel[grid](
        in_2, in_3, weight, bias, out,
        n_rows, n_cols,
        1e-7,
        in_2.stride(1), in_2.stride(2),
        in_3.stride(1), in_3.stride(2),
        out.stride(1), out.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_layernorm