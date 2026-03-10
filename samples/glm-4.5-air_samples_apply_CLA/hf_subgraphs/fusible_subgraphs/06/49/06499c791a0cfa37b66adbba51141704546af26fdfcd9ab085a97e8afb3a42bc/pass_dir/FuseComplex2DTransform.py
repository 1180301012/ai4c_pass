import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match pattern: (batch, groups, K, M) @ (batch, groups, M, N) -> view(batch, groups*K, spatial_x, spatial_y)"""
    tmp_0 = torch.matmul(in_1, in_0)
    # Extract reshape parameters from the view call in the original model
    spatial_dim = int(tmp_0.shape[-1] ** 0.5)  # Assume square spatial dimensions
    tmp_1 = tmp_0.view(tmp_0.shape[0], tmp_0.shape[1] * tmp_0.shape[2], spatial_dim, spatial_dim)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def complex_2d_transform_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size: tl.constexpr,
    groups: tl.constexpr,
    k_size: tl.constexpr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    spatial_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Optimized kernel for (batch, groups, K, M) @ (batch, groups, M, N) -> (batch, groups*K, spatial_x, spatial_y)"""
    # Each program handles one output element
    pid_batch = tl.program_id(0)
    pid_output_k = tl.program_id(1)
    pid_spatial_y = tl.program_id(2)
    pid_spatial_x = tl.program_id(3)
    
    # Boundary checks
    if pid_batch >= batch_size or pid_output_k >= groups * k_size or pid_spatial_y >= spatial_dim or pid_spatial_x >= spatial_dim:
        return
    
    # Map output K to groups and local K
    group_id = pid_output_k // k_size
    local_k_idx = pid_output_k % k_size
    n_idx = pid_spatial_y * spatial_dim + pid_spatial_x
    
    # Initialize accumulator
    acc = 0.0
    
    # Vectorized loop over M dimension (reduction)
    for m_idx in range(0, m_size, BLOCK_SIZE_K):
        m_offsets = m_idx + tl.arange(0, BLOCK_SIZE_K)
        mask_m = m_offsets < m_size
        
        # Load elements: x[pid_batch, group_id, local_k_idx, m_offsets] 
        # and y[pid_batch, group_id, m_offsets, n_idx]
        x_ptrs = x_ptr + (pid_batch * groups * k_size * m_size + group_id * k_size * m_size + local_k_idx * m_size + m_offsets)
        y_ptrs = y_ptr + (pid_batch * groups * m_size * n_size + group_id * m_size * n_size + m_offsets * n_size + n_idx)
        
        x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)
        y_vals = tl.load(y_ptrs, mask=mask_m, other=0.0)
        
        # Vectorized dot product
        acc += tl.dot(x_vals, y_vals)
    
    # Store result in [batch, groups*K, spatial_dim, spatial_dim] layout
    out_ptr = out_ptr + (pid_batch * (groups * k_size) * spatial_dim * spatial_dim + pid_output_k * spatial_dim * spatial_dim + pid_spatial_y * spatial_dim + pid_spatial_x)
    tl.store(out_ptr, acc)

@torch.fx.wrap
def complex_2d_transform_kernel_wrapper(in_0, in_1):
    """Wrapper function for the optimized kernel"""
    # Get input shapes
    batch_size = in_1.shape[0]
    groups = in_1.shape[1]
    k_size = in_1.shape[2]
    m_size = in_1.shape[3]
    n_size = in_0.shape[3]
    
    # Compute spatial dimensions (assuming square)
    spatial_dim = int(n_size ** 0.5)
    
    # Create output tensor
    out = torch.empty((batch_size, groups * k_size, spatial_dim, spatial_dim), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with 4D grid: (batch, groups*K, spatial_y, spatial_x)
    BLOCK_SIZE_M = 16  # Block size for K dimension
    BLOCK_SIZE_N = 16  # Block size for N dimension  
    BLOCK_SIZE_K = 32  # Block size for M dimension (reduction)
    
    grid = (batch_size, groups * k_size, spatial_dim, spatial_dim)
    
    complex_2d_transform_kernel[grid](
        in_0, in_1, out,
        batch_size, groups, k_size, m_size, n_size, spatial_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return complex_2d_transform_kernel_wrapper