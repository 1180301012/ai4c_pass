import torch
import triton
import triton.language as tl

@triton.jit
def fused_arithmetic_kernel(
    in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    in_1_dims, in_2_dims, in_3_dims,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Load in_1 slice [1, 4096, 32, 512] -> treat as [4096, 32, 512]
    in_1_slice = tl.load(in_1_ptr + (m_offset * 32 * 512 + n_offset * 512), 
                        mask=(m_offset < in_1_dims[0]) & (n_offset < in_1_dims[1]),
                        other=0.0)
    
    # Load in_2 slice with broadcasting [1, 1, 32, 512] -> [32, 512]
    in_2_slice = tl.load(in_2_ptr + (n_offset * 512),
                        mask=(n_offset < in_2_dims[1]),
                        other=0.0)
    
    # Load in_3 slice with broadcasting [1, 1, 32] -> [32]
    in_3_slice = tl.load(in_3_ptr + m_offset,
                        mask=(m_offset < in_3_dims[0]),
                        other=0.0)
    
    # Arithmetic sequence: (in_1 - in_2)^2 * in_3
    diff = in_1_slice - in_2_slice
    squared = diff * diff
    summed = tl.sum(squared, axis=0)  # Sum along last dimension
    weighted = summed * in_3_slice
    
    # Apply softmax
    max_val = tl.max(weighted)
    exp_val = tl.exp(weighted - max_val)
    sum_exp = tl.sum(exp_val)
    softmax_result = exp_val / sum_exp
    
    # Store result [1, 4096, 32] -> treat as [4096, 32]
    tl.store(out_ptr + (m_offset * in_3_dims[0] + n_offset), softmax_result,
             mask=(m_offset < in_1_dims[0]) & (n_offset < in_1_dims[1]))

@torch.fx.wrap
def fused_arithmetic_ops(in_1, in_2, in_3):
    # Input shapes: in_1[1,4096,32,512], in_2[1,1,32,512], in_3[1,1,32]
    out_shape = [1, 4096, 32]
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_M = 32  # Process 32 rows at a time
    BLOCK_SIZE_N = 32  # Process 32 columns at a time
    
    grid = lambda meta: (
        (out_shape[1] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (out_shape[2] + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        1
    )
    
    # Launch kernel
    fused_arithmetic_kernel[grid](
        in_1, in_2, in_3, out,
        [out_shape[1], out_shape[2]],  # in_1 dims [4096, 32]
        [out_shape[2]],                 # in_2 dims [32]
        [out_shape[1]],                 # in_3 dims [4096]
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def pattern(in_1, in_2, in_3):
    """Pattern to match arithmetic sequence: (in_1 - in_2)^2.sum(dim=3) * in_3 + softmax(dim=2)"""
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    tmp_4 = in_3 * tmp_3
    out = torch.nn.functional.softmax(tmp_4, dim = 2)
    return out

def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)

def replacement_func():
    return fused_arithmetic_ops