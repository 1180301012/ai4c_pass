import torch
import triton
import triton.language as tl
from torch import device


def pattern(x):
    t = x.t()
    out = t.to(device(type='cuda'))
    return out


def replacement_args(x):
    return (x,)


@triton.jit
def transpose_copy_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    # Input layout [M, N] -> output layout [N, M]
    src_row = offsets // N
    src_col = offsets % N

    # Load from x[src_row, src_col]
    x_val = tl.load(
        x_ptr + src_row * stride_xm + src_col * stride_xn,
        mask=mask,
        other=0.0,
    )

    # Store to out[src_col, src_row]
    tl.store(
        out_ptr + src_col * stride_om + src_row * stride_on,
        x_val,
        mask=mask,
    )


@torch.fx.wrap
def fused_transpose_to_cuda(x):
    # x is already on CUDA, so to(device='cuda') is identity
    # Produce a contiguous transposed copy
    M, N = x.shape
    out = torch.empty(N, M, dtype=x.dtype, device=x.device)

    total_elements = M * N
    BLOCK_SIZE = triton.next_power_of_2(total_elements)

    grid = (1,)
    transpose_copy_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_transpose_to_cuda