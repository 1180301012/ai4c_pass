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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    ],
    key=['D'],
)
@triton.jit
def fused_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    stride_in2, stride_in3, stride_out,
    N, D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    in_2_row_ptr = in_2_ptr + row_idx * stride_in2
    in_3_row_ptr = in_3_ptr + row_idx * stride_in3
    out_row_ptr = out_ptr + row_idx * stride_out

    # First pass: compute mean and sum of squares
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, D, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(in_2_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(in_3_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vals = x + y
        _sum += vals
        _sum_sq += vals * vals
    mean = tl.sum(_sum, axis=0) / D
    var = tl.sum(_sum_sq, axis=0) / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for start in range(0, D, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(in_2_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(in_3_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vals = x + y
        norm = (vals - mean) * rstd

        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        out_val = norm * w + b
        tl.store(out_row_ptr + cols, out_val, mask=mask)


@torch.fx.wrap
def fused_layernorm(in_0, in_1, in_2, in_3):
    # Get dimensions
    D = in_2.shape[-1]
    # Compute N as product of all dimensions except last
    # For 3D [B, S, D]: N = B * S; for 2D [N, D]: N = N
    if len(in_2.shape) == 3:
        N = in_2.shape[0] * in_2.shape[1]
    else:
        N = in_2.shape[0]

    # Create output with same shape as input, but float32 dtype
    out = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)

    # Strides for row-wise access
    # For contiguous tensor, stride(-2) gives the row stride
    stride_in2 = in_2.stride(-2)
    stride_in3 = in_3.stride(-2)
    stride_out = out.stride(-2)

    grid = (N,)

    # BLOCK_SIZE is handled by autotune - do not pass it explicitly
    fused_layernorm_kernel[grid](
        in_2, in_3, in_1, in_0, out,
        stride_in2, stride_in3, stride_out,
        N, D,
        1e-07,
    )

    return (out,)


def replacement_func():
    return fused_layernorm