import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Match matmul + scale pattern"""
    # Use torch.matmul exactly like the model
    mm_result = torch.matmul(in_2, in_1)
    scaled = mm_result * in_0
    return scaled


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement"""
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scale_kernel(
    in_2_ptr, in_1_ptr, scale_ptr,
    out_ptr,
    M: tl.constexpr, K: tl.constexpr,
    stride_in2_m, stride_in2_k,
    stride_in1_k,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized fused matmul + scale kernel"""
    row_idx = tl.program_id(0)
    
    # Accumulator for dot product
    accum = 0.0
    
    # Load scale value (scalar tensor)
    scale_val = tl.load(scale_ptr)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE):
        offs_k = k + tl.arange(0, BLOCK_SIZE)
        mask_k = offs_k < K
        
        # Load from in_2: [M, K] matrix
        offs_in2 = row_idx * stride_in2_m + offs_k * stride_in2_k
        in_2_vals = tl.load(in_2_ptr + offs_in2, mask=mask_k, other=0.0)
        
        # Load from in_1: [K, 1] matrix - only column 0
        offs_in1 = offs_k * stride_in1_k
        in_1_vals = tl.load(in_1_ptr + offs_in1, mask=mask_k, other=0.0)
        
        # Compute partial dot product
        accum += tl.sum(in_2_vals * in_1_vals)
    
    # Apply scaling
    result = accum * scale_val
    
    # Store result
    tl.store(out_ptr + row_idx, result)


@torch.fx.wrap
def optimized_matmul_scale(in_0, in_1, in_2):
    """
    Fused matmul + scale using Triton kernel
    
    in_0: scalar tensor (logit_scale)
    in_1: [1024, 1] matrix
    in_2: [2, 1024] matrix
    Returns: scaled_matmul [2, 1]
    """
    M, K = in_2.shape  # [2, 1024]
    
    # Allocate output tensor - using torch.empty (allowed API)
    out = torch.empty((M, 1), dtype=in_2.dtype, device=in_2.device)
    
    # Launch Triton kernel - pass tensor directly, load inside kernel
    grid = (M,)
    BLOCK_SIZE = 1024
    
    fused_matmul_scale_kernel[grid](
        in_2, in_1, in_0,
        out,
        M, K,
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0),
        BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return optimized_matmul_scale