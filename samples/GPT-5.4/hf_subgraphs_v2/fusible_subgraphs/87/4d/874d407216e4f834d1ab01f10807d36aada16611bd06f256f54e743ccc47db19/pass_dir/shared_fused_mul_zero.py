import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    ],
    key=["n_rows", "n_cols", "zero_rows"],
)
@triton.jit
def fused_mul_zero_kernel(
    weight_ptr,
    x_ptr,
    out_mul_ptr,
    out_zero_ptr,
    n_rows,
    n_cols,
    zero_rows,
    stride_x0,
    stride_x1,
    stride_mul0,
    stride_mul1,
    stride_zero0,
    stride_zero1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_x0 + offs_n[None, :] * stride_x1
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs_m, mask=mask_m, other=0.0)
    out = x * w[:, None]

    mul_ptrs = out_mul_ptr + offs_m[:, None] * stride_mul0 + offs_n[None, :] * stride_mul1
    tl.store(mul_ptrs, out, mask=mask)

    zero_mask_m = offs_m < zero_rows
    zero_mask = zero_mask_m[:, None] & mask_n[None, :]
    zero_ptrs = out_zero_ptr + offs_m[:, None] * stride_zero0 + offs_n[None, :] * stride_zero1
    tl.store(zero_ptrs, 0.0, mask=zero_mask)


@torch.fx.wrap
def fused_mul_zero_dispatch(in_1, in_2, route):
    shape = in_2.size()
    n_rows = shape[0]
    n_cols = shape[1]
    strides = in_2.stride()
    stride_x0 = strides[0]
    stride_x1 = strides[1]

    out_mul = torch.empty_like(in_2)
    out_mul_strides = out_mul.stride()
    stride_mul0 = out_mul_strides[0]
    stride_mul1 = out_mul_strides[1]

    if route == "gae":
        zero_rows = 1000
        zero_shape = (1000, 16)
    elif route == "rect":
        zero_rows = 128
        zero_shape = (128, 128)
    else:
        zero_rows = n_rows
        zero_shape = (n_rows, n_cols)

    out_zero = torch.empty(zero_shape, dtype=in_2.dtype, device=in_2.device)
    out_zero_strides = out_zero.stride()
    stride_zero0 = out_zero_strides[0]
    stride_zero1 = out_zero_strides[1]

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]), triton.cdiv(n_cols, meta["BLOCK_N"]))
    fused_mul_zero_kernel[grid](
        weight_ptr=in_1,
        x_ptr=in_2,
        out_mul_ptr=out_mul,
        out_zero_ptr=out_zero,
        n_rows=n_rows,
        n_cols=n_cols,
        zero_rows=zero_rows,
        stride_x0=stride_x0,
        stride_x1=stride_x1,
        stride_mul0=stride_mul0,
        stride_mul1=stride_mul1,
        stride_zero0=stride_zero0,
        stride_zero1=stride_zero1,
    )
    return out_mul, out_zero


def shared_replacement_func():
    return fused_mul_zero_dispatch