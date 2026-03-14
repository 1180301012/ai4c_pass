import torch
import triton
import triton.language as tl


# Define autotune configurations for different hidden dimension sizes
@triton.autotune(
    configs=[
        # Configs for different N sizes (hidden dimension)
        # Each config: (num_warps, num_stages, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_N': 256}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE_N': 128}, num_warps=2, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def fused_add_mul_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_in_2_0, stride_in_2_1, stride_in_2_2,
    stride_in_3_0, stride_in_3_1, stride_in_3_2,
    stride_out_0, stride_out_1, stride_out_2,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for: ((in_3 + in_2) * in_1) + in_0
    
    in_0, in_1: 1D tensors [N]
    in_2, in_3: 3D tensors [M, K, N]
    Output: 3D tensor [M, K, N]
    
    Grid: M programs, each processes all K*N elements for one M
    """
    # Each program handles one M dimension (batch element)
    pid = tl.program_id(0)
    
    if pid >= M:
        return
    
    # Load bias and weight (1D, size N) - broadcast across all K
    bias = tl.load(in_0_ptr + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N)
    weight = tl.load(in_1_ptr + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N)
    
    # Process all K*N elements for this M
    # Each program processes a contiguous block of K elements, looping over N
    for k_idx in range(K):
        # Compute offsets for in_2 and in_3
        in_2_offset = pid * stride_in_2_0 + k_idx * stride_in_2_1 + tl.arange(0, BLOCK_SIZE_N)
        in_3_offset = pid * stride_in_3_0 + k_idx * stride_in_3_1 + tl.arange(0, BLOCK_SIZE_N)
        
        mask = tl.arange(0, BLOCK_SIZE_N) < N
        
        in_2_vals = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
        in_3_vals = tl.load(in_3_ptr + in_3_offset, mask=mask, other=0.0)
        
        # Compute fused operation
        result = (in_3_vals + in_2_vals) * weight + bias
        
        # Store result
        out_offset = pid * stride_out_0 + k_idx * stride_out_1 + tl.arange(0, BLOCK_SIZE_N)
        tl.store(out_ptr + out_offset, result, mask=mask)


def pattern(in_0, in_1, in_2, in_3):
    """Match the pattern: ((in_3 + in_2) * in_1) + in_0"""
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Fused kernel that computes: ((in_3 + in_2) * in_1) + in_0
    Returns the fused output (the slice will be computed separately)
    """
    # Determine tensor shapes
    in_2_shape = in_2.shape
    
    if len(in_2_shape) == 3:
        M = in_2_shape[0]  # Batch dimension
        K = in_2_shape[1]  # Sequence or other dimension
        N = in_2_shape[2]  # Hidden dimension (should match in_0 size)
        
        # Output shape is same as in_2
        output = torch.empty_like(in_2)
        
        # Grid: M programs (each handles one batch element)
        grid = (M,)
        
        fused_add_mul_add_kernel[grid](
            in_0, in_1, in_2, in_3,
            output,
            M, K, N,
            in_2.stride(0), in_2.stride(1), in_2.stride(2),
            in_3.stride(0), in_3.stride(1), in_3.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )
    elif len(in_2_shape) == 2:
        M = in_2_shape[0]
        K = 1  # No middle dimension for 2D
        N = in_2_shape[1]
        
        output = torch.empty_like(in_2)
        
        grid = (M,)
        
        fused_add_mul_add_kernel[grid](
            in_0, in_1, in_2, in_3,
            output,
            M, K, N,
            in_2.stride(0), 0, in_2.stride(1),
            in_3.stride(0), 0, in_3.stride(1),
            output.stride(0), 0, output.stride(1),
        )
    else:
        # Fallback for other cases
        tmp_2 = in_3 + in_2
        tmp_3 = tmp_2 * in_1
        tmp_4 = tmp_3 + in_0
        return tmp_4
    
    return output


def replacement_func():
    return kernel_wrapper