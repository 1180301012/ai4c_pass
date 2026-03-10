import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match pattern: Matrix multiplication followed by view operation that squeezes dimensions"""
    # Handle both torch.matmul and @ operator
    tmp_0 = torch.matmul(in_1, in_0)  # This should match @ operator too due to overloading
    # Handle various view patterns that reduce dimensions
    # Common patterns: view(batch, K, 1, 1) or similar dimension reduction
    result_shape = tmp_0.shape
    if len(result_shape) == 4:
        # Try to match patterns where we're squeezing dimensions
        # For example: [batch, 1, K, 1] -> [batch, K, 1, 1]
        tmp_1 = tmp_0.view(result_shape[0], result_shape[2], 1, 1)
        return tmp_1
    else:
        # Handle other possible patterns by computing a suitable view
        # This is a more general approach
        if len(result_shape) >= 2:
            # Try to create a 4D output by keeping first two dims and making last two 1x1
            view_shape = list(result_shape[:2]) + [1, 1]
            if len(view_shape) < 4:
                view_shape += [1] * (4 - len(view_shape))
            tmp_1 = tmp_0.view(*view_shape[:4])
            return tmp_1
    return tmp_0  # fallback

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def linear_transform_squeeze_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size: tl.constexpr,
    k_size: tl.constexpr,
    m_size: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Optimized kernel for (batch, 1, K, M) @ (batch, 1, M, 1) -> (batch, K, 1, 1)"""
    # Program IDs: one program per batch element
    pid = tl.program_id(0)
    
    # Thread indices within the program
    k_idx = tl.program_id(1)
    
    # Boundary checks
    if pid >= batch_size or k_idx >= k_size:
        return
    
    # Initialize output
    acc = 0.0
    
    # Vectorized loop over M dimension (reduction)
    for m_idx in range(0, m_size, BLOCK_SIZE_K):
        # Create vector of M indices
        m_offsets = m_idx + tl.arange(0, BLOCK_SIZE_K)
        mask_m = m_offsets < m_size
        
        # Load elements: x[pid, 0, k_idx, m_offsets] and y[pid, 0, m_offsets, 0]
        x_ptrs = x_ptr + (pid * 1 * k_size * m_size + 0 * k_size * m_size + k_idx * m_size + m_offsets)
        y_ptrs = y_ptr + (pid * 1 * m_size * 1 + 0 * m_size * 1 + m_offsets * 1 + 0)
        
        x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)
        y_vals = tl.load(y_ptrs, mask=mask_m, other=0.0)
        
        # Vectorized dot product
        acc += tl.dot(x_vals, y_vals)
    
    # Store result directly in [batch, K, 1, 1] layout
    out_ptr = out_ptr + (pid * k_size * 1 * 1 + k_idx * 1 * 1 + 0 * 1 + 0)
    tl.store(out_ptr, acc)

@torch.fx.wrap
def linear_transform_squeeze_kernel_wrapper(in_0, in_1):
    """Wrapper function for the optimized kernel"""
    # Get input shapes
    batch_size = in_1.shape[0]
    k_size = in_1.shape[2]
    m_size = in_1.shape[3]
    
    # Create output tensor
    out = torch.empty((batch_size, k_size, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with 2D grid: (batch, K)
    BLOCK_SIZE_K = 32  # Block size for M dimension (reduction)
    
    # Grid dimensions: (batch_size, k_size)
    grid = (batch_size, k_size)
    
    linear_transform_squeeze_kernel[grid](
        in_0, in_1, out,
        batch_size, k_size, m_size,
        BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return linear_transform_squeeze_kernel_wrapper