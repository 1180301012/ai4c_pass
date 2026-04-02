import torch
import triton
import triton.language as tl

def pattern(x):
    """Test mean pattern - match actual operation in the computation"""
    return x.mean(dim=-2, keepdim=True)

def replacement_args(x):
    return (x,)

@triton.jit
def test_mean_kernel(
    x_ptr,        # Input tensor [N, D2, D3]
    out_ptr,      # Output tensor [N, 1, D3]
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
    
    # Compute mean along dimension 1 (D2 size)
    sum_val = 0.0
    elements_processed = 0
    
    for k in range(0, D2, BLOCK_SIZE_M):
        end_k = min(k + BLOCK_SIZE_M, D2)
        for i in range(k, end_k):
            offset = m * D2 * D3 + i * D3 + n
            val = tl.load(x_ptr + offset)
            sum_val += val
            elements_processed += 1
    
    # Compute mean (avoid division by zero for robustness)
    mean_val = sum_val / max(elements_processed, 1)
    
    # Store result
    output_offset = m * 1 * D3 + n
    tl.store(out_ptr + output_offset, mean_val)

@torch.fx.wrap
def test_triton_mean(x):
    """Optimized mean operation along dim=-2"""
    N, D2, D3 = x.shape
    
    # Create output tensor [N, 1, D3]
    output = torch.empty((N, 1, D3), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 256  # Block size for reduction dimension
    BLOCK_SIZE_N = 256  # Block size for feature dimension
    
    grid_m = N
    grid_n = (D3 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    test_mean_kernel[(grid_m, grid_n)](
        x,
        output,
        N,
        D2,
        D3,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return test_triton_mean