import torch
import triton
import triton.language as tl


# Pattern matching function - must exactly match the computation in model.py
def pattern(in_0, in_1, in_2):
    """
    Match the pattern:
    tmp_0 = torch.matmul(in_2, in_1)  # [2, 512] @ [512, 1] = [2, 1]
    tmp_1 = tmp_0 * in_0              # [2, 1] * scalar = [2, 1]
    tmp_2 = tmp_1.T                   # [1, 2]
    return (tmp_1, tmp_2)
    """
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    tmp_2 = tmp_1.T
    return (tmp_1, tmp_2)


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_dot_scale_kernel(
    in_2_ptr,  # [2, K] 
    in_1_ptr,  # [K, 1]
    scale_ptr, # scalar
    out_ptr,   # [2, 1]
    K: tl.constexpr,
):
    """
    Optimized single-kernel for 2 dot products.
    """
    # Each program handles one row
    row = tl.program_id(0)
    
    # Load weight column (shared by both computations)
    k_idx = tl.arange(0, K)
    weight = tl.load(in_1_ptr + k_idx)
    
    # Load input row
    input_row = tl.load(in_2_ptr + row * K + k_idx)
    
    # Dot product
    dot = tl.sum(input_row * weight, axis=0)
    
    # Scale
    scale = tl.load(scale_ptr)
    result = dot * scale
    
    # Store
    tl.store(out_ptr + row, result)


@torch.fx.wrap
def fused_matmul_scale_transpose_impl(in_0, in_1, in_2):
    """
    Fused implementation of matmul + scale + transpose.
    """
    M = in_2.shape[0]
    K = in_2.shape[1]
    N = in_1.shape[1]
    
    # Allocate output
    out = torch.empty((M, N), device=in_2.device, dtype=in_2.dtype)
    
    # Launch kernel
    fused_dot_scale_kernel[(M,)](
        in_2,
        in_1,
        in_0,
        out,
        K,
        num_warps=4,
    )
    
    # Transpose is a view
    out_t = out.T
    
    return out, out_t


def fused_matmul_scale_transpose(in_0, in_1, in_2):
    out, out_t = fused_matmul_scale_transpose_impl(in_0, in_1, in_2)
    return (out, out_t)


def replacement_func():
    return fused_matmul_scale_transpose