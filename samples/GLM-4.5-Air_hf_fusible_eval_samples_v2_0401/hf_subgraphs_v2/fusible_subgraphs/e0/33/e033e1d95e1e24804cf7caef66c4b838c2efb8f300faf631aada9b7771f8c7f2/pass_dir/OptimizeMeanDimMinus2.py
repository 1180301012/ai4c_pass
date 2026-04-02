import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Match Mean operation along dim=-2"""
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def optimized_mean_kernel(
    input_ptr,    # [N, 4096, 256]
    output_ptr,   # [N, 1, 256] 
    N,
    D2,
    D3,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized mean computation along dimension 1 (dim=-2)"""
    # Each program handles one (batch, feature) pair
    m = tl.program_id(0)  # batch dimension
    n = tl.program_id(1)  # feature dimension (D3)
    
    # Compute mean along dimension 1 (D2 size = 4096)
    sum_val = 0.0
    for k in range(0, D2, BLOCK_SIZE_M):
        # Compute bounds for current block
        end_k = min(k + BLOCK_SIZE_M, D2)
        
        # Load input values for current block
        for i in range(k, end_k):
            offset = m * D2 * D3 + i * D3 + n
            val = tl.load(input_ptr + offset)
            sum_val += val
    
    # Compute mean
    mean_val = sum_val / D2
    
    # Store result
    output_offset = m * 1 * D3 + n
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap  
def optimized_mean(in_2):
    """Optimized mean operation along dim=-2"""
    N, D2, D3 = in_2.shape
    
    # For very small tensors, use built-in mean to avoid Triton overhead
    total_elements = N * D2 * D3
    if total_elements < 2048:  # Small tensor threshold
        return in_2.mean(dim=-2, keepdim=True)
    
    # Create output tensor [N, 1, D3]
    output = torch.empty((N, 1, D3), dtype=in_2.dtype, device=in_2.device)
    
    # Adaptive block sizes based on tensor characteristics
    BLOCK_SIZE_M = min(256, D2, max(64, D2 // 8))  # Adaptive for reduction
    BLOCK_SIZE_N = min(256, max(32, D3 // 8))     # Adaptive for features
    
    # Special handling for different tensor sizes
    if D3 < 128:
        BLOCK_SIZE_N = min(128, D3)  # Use smaller blocks for few features
    elif D3 > 2048:
        BLOCK_SIZE_N = min(512, max(128, D3 // 16))  # Larger blocks for many features
    
    # Calculate grid dimensions
    grid_m = N
    grid_n = (D3 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_mean_kernel[(grid_m, grid_n)](
        in_2,
        output,
        N,
        D2,
        D3,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return optimized_mean