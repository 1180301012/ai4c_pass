"""
Optimization pass for rtpose-l style computation using Triton kernels.
"""

import torch
import triton
import triton.language as tl


# Triton kernel for element-wise multiply with broadcasting
@triton.jit
def mul_broadcast_kernel(
    x_ptr, scale_ptr, out_ptr,
    B, M, N,
    stride_x0, stride_x1, stride_x2,
    stride_s0,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise multiply: x * scale (scale broadcasts over leading dims)
    x shape: [B, M, N], scale shape: [N], out shape: [B, M, N]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < B * M * N
    
    # Calculate indices from flat offset
    b_idx = offs // (M * N)
    rem = offs % (M * N)
    m_idx = rem // N
    n_idx = rem % N
    
    x = tl.load(x_ptr + b_idx * stride_x0 + m_idx * stride_x1 + n_idx * stride_x2, mask=mask, other=0.0)
    # scale broadcasts over B and M dims, scale has shape [N] with stride_s0
    scale = tl.load(scale_ptr + n_idx * stride_s0, mask=(n_idx < N), other=0.0)
    
    out = x * scale
    tl.store(out_ptr + b_idx * stride_x0 + m_idx * stride_x1 + n_idx * stride_x2, out, mask=mask)


@torch.fx.wrap
def triton_mul_broadcast(x, scale):
    """
    Triton wrapper for element-wise multiply with broadcasting.
    x: [B, M, N], scale: [N] (broadcasts over B and M)
    """
    B, M, N = x.shape
    BLOCK_SIZE = 1024
    num_programs = (B * M * N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    mul_broadcast_kernel[(num_programs,)](
        x, scale, output,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        scale.stride(0),
        BLOCK_SIZE,
    )
    return output


def pattern(x, y):
    """
    Match element-wise multiply pattern.
    """
    return x * y


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return mul_broadcast_replacement


def mul_broadcast_replacement(x, y):
    """
    Replacement function using Triton kernel.
    """
    return triton_mul_broadcast(x, y)