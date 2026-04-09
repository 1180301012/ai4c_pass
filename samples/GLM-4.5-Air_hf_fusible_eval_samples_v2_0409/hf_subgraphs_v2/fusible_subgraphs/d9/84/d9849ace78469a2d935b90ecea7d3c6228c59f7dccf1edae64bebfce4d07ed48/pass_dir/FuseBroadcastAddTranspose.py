import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern matches: transpose(1, 2) operation
    """
    # Match transpose only (this was working before)
    in_2 = in_0
    tmp_2 = in_2.transpose(1, 2)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,       # Input tensor: [1, K, N] -> [1, 128, 19] 
    output_ptr,      # Output tensor: [1, N, K] -> [1, 19, 128]
    N: tl.constexpr,  # 19 (N dimension)
    TOTAL_ELEMENTS: tl.constexpr,
):
    """Optimized transpose kernel for [1, K, N] -> [1, N, K]"""
    # Each thread handles one element
    linear_idx = tl.program_id(0)
    
    # Check bounds to avoid out-of-memory access
    if linear_idx >= TOTAL_ELEMENTS:
        return
    
    # Convert linear index to 2D indices
    # For input [1, K, N]: assume row-major layout so K (fast changing), N (slow changing)
    k_idx = linear_idx // N   # Row index in K dimension  
    n_idx = linear_idx % N     # Column index in N dimension
    
    # Calculate proper offsets
    # Input [1, K, N]: element at [0, k_idx, n_idx] -> offset = k_idx * N + n_idx
    input_offset = k_idx * N + n_idx
    
    # Output [1, N, K]: element at [0, n_idx, k_idx] -> offset = n_idx * K + k_idx  
    # Note: K needs to be passed from host to kernel
    output_offset = n_idx * (TOTAL_ELEMENTS // N) + k_idx
    
    # Load and store
    val = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, val)

@torch.fx.wrap  
def optimized_transpose(a):
    """Optimized transpose kernel wrapper for [1, K, N] -> [1, N, K] transpose"""
    _, K, N = a.shape  # a: [1, K, N] -> [1, 128, 19]
    
    output_shape = (1, N, K)  # [1, N, K] -> [1, 19, 128]
    output = torch.empty(output_shape, dtype=a.dtype, device=a.device)
    
    # Total elements to process (excluding first dim=1)
    total_elements = K * N
    
    # Use 1D grid where each thread handles exactly one element
    grid_size = total_elements
    
    # Launch kernel with 1D grid - wrap in tuple
    optimized_transpose_kernel[(grid_size,)](
        input_ptr=a,
        output_ptr=output,
        N=N,
        TOTAL_ELEMENTS=total_elements
    )
    
    return output

def replacement_func():
    return optimized_transpose