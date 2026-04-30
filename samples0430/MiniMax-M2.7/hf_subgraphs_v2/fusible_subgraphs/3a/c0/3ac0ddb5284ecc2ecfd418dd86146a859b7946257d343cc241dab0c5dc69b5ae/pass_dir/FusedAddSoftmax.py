import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, H, M, N,
    stride_in_0_0, stride_in_0_1, stride_in_0_2, stride_in_0_3,
    stride_in_1_0, stride_in_1_1, stride_in_1_2, stride_in_1_3,
    stride_out_0, stride_out_1, stride_out_2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: add + softmax
    - Add in_1 [B, H, M, N] + in_0 [1, 1, M, N] (broadcasting)
    - Softmax on dim=-1
    - Output [B*H, M, N]
    """
    # Each program handles one row of the softmax: position (b, h, m)
    # Total programs = B * H * M
    pid = tl.program_id(0)
    
    # Compute b, h, m from pid
    b = pid // (H * M)
    h = (pid // M) % H
    m = pid % M
    
    # Compute row offset for the last dimension (N elements)
    row_offset_in_0 = m * stride_in_0_2
    row_offset_in_1 = b * stride_in_1_0 + h * stride_in_1_1 + m * stride_in_1_2
    
    # Compute offsets for loading N elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data - each thread loads its portion of the row
    in_0_vals = tl.load(in_0_ptr + row_offset_in_0 + offsets * stride_in_0_3, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + row_offset_in_1 + offsets * stride_in_1_3, mask=mask, other=0.0)
    
    # Add with broadcasting
    summed = in_0_vals + in_1_vals
    
    # Softmax computation with numerical stability
    max_val = tl.max(summed)
    max_vals = tl.expand_dims(max_val, 0)
    exp_vals = tl.exp(summed - max_vals)
    sum_exp = tl.sum(exp_vals)
    sum_exps = tl.expand_dims(sum_exp, 0)
    softmax_vals = exp_vals / sum_exps
    
    # Store output [B*H, M, N]
    out_offset = (b * H + h) * stride_out_0 + m * stride_out_1 + offsets * stride_out_2
    tl.store(out_ptr + out_offset, softmax_vals, mask=mask)


@triton.jit
def fused_add_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    total_elements: tl.constexpr,
    stride_in_0_0, stride_in_0_1, stride_in_0_2, stride_in_0_3,
    stride_in_1_0, stride_in_1_1, stride_in_1_2, stride_in_1_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    B, H, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for element-wise add with broadcasting.
    Each program handles BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute b, h, m, n indices
    n = offsets % N
    temp = offsets // N
    m = temp % M
    temp = temp // M
    h = temp % H
    b = temp // H
    
    # Compute offsets for in_0 and in_1 with broadcasting
    # in_0 has shape [1, 1, M, N], broadcast to [B, H, M, N]
    offset_in_0 = b * stride_in_0_0 + h * stride_in_0_1 + m * stride_in_0_2 + n * stride_in_0_3
    offset_in_1 = b * stride_in_1_0 + h * stride_in_1_1 + m * stride_in_1_2 + n * stride_in_1_3
    
    # Load values
    in_0_val = tl.load(in_0_ptr + offset_in_0, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offset_in_1, mask=mask, other=0.0)
    
    # Compute sum
    result = in_0_val + in_1_val
    
    # Store result
    offset_out = b * stride_out_0 + h * stride_out_1 + m * stride_out_2 + n * stride_out_3
    tl.store(out_ptr + offset_out, result, mask=mask)


# Use a simple, optimized kernel without autotuning for minimal overhead
@triton.jit
def fused_add_kernel_opt(
    in_0_ptr, in_1_ptr, out_ptr,
    total_elements,
    stride_in_0_0, stride_in_0_1, stride_in_0_2, stride_in_0_3,
    stride_in_1_0, stride_in_1_1, stride_in_1_2, stride_in_1_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    B, H, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    n = offsets % N
    temp = offsets // N
    m = temp % M
    temp = temp // M
    h = temp % H
    b = temp // H
    
    offset_in_0 = b * stride_in_0_0 + h * stride_in_0_1 + m * stride_in_0_2 + n * stride_in_0_3
    offset_in_1 = b * stride_in_1_0 + h * stride_in_1_1 + m * stride_in_1_2 + n * stride_in_1_3
    
    in_0_val = tl.load(in_0_ptr + offset_in_0, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offset_in_1, mask=mask, other=0.0)
    
    result = in_0_val + in_1_val
    
    offset_out = b * stride_out_0 + h * stride_out_1 + m * stride_out_2 + n * stride_out_3
    tl.store(out_ptr + offset_out, result, mask=mask)


@torch.fx.wrap
def fused_add_softmax_wrapper(in_0, in_1):
    """
    Wrapper function that launches an optimized fused add kernel.
    
    Args:
        in_0: Tensor [1, 1, M, N] - attention mask (broadcasts)
        in_1: Tensor [B, H, M, N] - the main input
    
    Returns:
        out: Tensor with same shape as in_1
    """
    B, H, M, N = in_1.shape
    total_elements = B * H * M * N
    
    # Allocate output with same shape as in_1
    out = torch.empty_like(in_1)
    
    # Use a block size that divides well into our problem sizes
    BLOCK_SIZE = 512
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_kernel_opt[(num_programs,)](
        in_0, in_1, out,
        total_elements,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """Match add operation with broadcasting"""
    return in_1 + in_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_softmax_wrapper