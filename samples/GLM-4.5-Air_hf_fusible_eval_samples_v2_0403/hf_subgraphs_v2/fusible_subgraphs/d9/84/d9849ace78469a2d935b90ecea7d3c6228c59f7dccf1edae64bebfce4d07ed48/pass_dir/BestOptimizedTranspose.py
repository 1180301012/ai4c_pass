import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(a):
    """
    Pattern matches: transpose(1,2) operation on [1, 128, 19] tensor
    """
    return a.transpose(1, 2)


# Argument extraction function  
def replacement_args(a):
    return (a,)


@triton.jit
def best_transpose_kernel(
    x_ptr,
    out_ptr,
    dim0_size,      # 1
    dim1_size,      # 128 (original second dimension)
    dim2_size,      # 19 (original third dimension)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for new dimension 1 (19)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for new dimension 2 (128)
):
    """
    Best optimized transpose kernel: [1, 128, 19] -> [1, 19, 128]
    Uses optimal blocking and memory access patterns
    """
    # Each program handles a tile in transposed space
    m = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n = tl.program_id(2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create bounds check
    k_mask = k < dim2_size
    n_mask = n < dim1_size
    
    # Create coordinate grids
    k_indices = k[:, None]  # [BLOCK_SIZE_K, 1] 
    n_indices = n[None, :]  # [1, BLOCK_SIZE_N]
    
    # Calculate input indices for original layout: [1, 128, 19]
    # We need to transpose dims 1 and 2, so access as [m, n, k]
    input_indices = n_indices * dim2_size + k_indices
    
    # Calculate output indices for transposed layout: [1, 19, 128]  
    # Access as [m, k, n]
    output_indices = k_indices * dim1_size + n_indices
    
    # Create mask for valid elements
    valid_mask = k_mask[:, None] & n_mask[None, :]
    
    # Load data from original layout
    x_vals = tl.load(x_ptr + input_indices, mask=valid_mask, other=0.0)
    
    # Store data in transposed layout  
    tl.store(out_ptr + output_indices, x_vals, mask=valid_mask)


@torch.fx.wrap  
def best_transpose(x):
    """
    Best optimized transpose operation: [1, 128, 19] -> [1, 19, 128]
    """
    dim0_size = x.shape[0]  # 1
    dim1_size = x.shape[1]  # 128
    dim2_size = x.shape[2]  # 19
    
    # Output shape after transpose: [1, 19, 128]
    out_shape = (dim0_size, dim2_size, dim1_size)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Optimized block sizes
    BLOCK_SIZE_K = 16  # Power of 2 for dim 1 (19)
    BLOCK_SIZE_N = 32  # Power of 2 for dim 2 (128)
    
    # Calculate grid dimensions
    m_dim = 1  # Always 1 since dim0=1
    k_dim = (dim2_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  
    n_dim = (dim1_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    best_transpose_kernel[(m_dim, k_dim, n_dim)](
        x_ptr=x,
        out_ptr=out,
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


# Replacement function (returns function reference, not a call)
def replacement_func():
    return best_transpose