import torch
import triton
import triton.language as tl

def pattern(x, divisor):
    """Pattern that matches scalar division followed by transpose(-1, -2)"""
    tmp_0 = x / divisor
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

def replacement_args(x, divisor):
    """Extract arguments needed for the replacement"""
    return (x, divisor)

@triton.jit
def fused_div_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    divisor,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized kernel that fuses scalar division and transpose operations"""
    
    # Program ids for batch and flattened matrix
    b = tl.program_id(0)
    h = tl.program_id(1) 
    flat_idx = tl.program_id(2)
    
    # Unflatten to get block coordinates
    n_block = flat_idx // grid_m
    m_block = flat_idx % grid_m
    
    # Calculate element coordinates within block (vectorized)
    m_coords = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_coords = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Broadcast to create coordinate matrix
    m_matrix = m_coords[:, None]  # Shape: (BLOCK_SIZE_M, 1)
    n_matrix = n_coords[None, :]  # Shape: (1, BLOCK_SIZE_N)
    
    # Calculate offset for current batch/head
    base_offset = (b * num_heads + h) * seq_len * head_dim
    
    # Calculate input and output offsets (vectorized)
    input_offsets = base_offset + m_matrix * head_dim + n_matrix
    output_offsets = base_offset + n_matrix * seq_len + m_matrix
    
    # Create masks for valid coordinates
    m_mask = m_matrix < seq_len
    n_mask = n_matrix < head_dim
    mask = m_mask & n_mask  # Element-wise for the matrix
    
    # Load input data, apply division, store transposed result
    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    out = x / divisor
    tl.store(out_ptr + output_offsets, out, mask=mask)

@torch.fx.wrap
def fused_div_transpose_kernel_wrapper(x, divisor):
    """Kernel wrapper to launch the fused operations"""
    
    # Get tensor shape info
    batch_size, num_heads, seq_len, head_dim = x.shape
    
    # For small tensors, still use Triton kernel but with different block sizes
    # to avoid launching too many threads
    
    # Always use Triton kernel with minimal overhead for these workloads
    batch_size = max(1, batch_size)
    num_heads = max(1, num_heads)
    
    total_elements = batch_size * num_heads * seq_len * head_dim
    
    # Use medium block sizes for best efficiency
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    
    # Calculate grid size - flattened 3D grid
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (batch_size, num_heads, grid_m * grid_n)
    
    # Create output tensor - transposed shape: (batch_size, num_heads, head_dim, seq_len)
    out = torch.empty((batch_size, num_heads, head_dim, seq_len), dtype=x.dtype, device=x.device)
    
    # Launch kernel with simpler approach
    fused_div_transpose_kernel[grid](
        x, out,
        batch_size, num_heads, seq_len, head_dim, divisor,
        grid_m, grid_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    """Returns the kernel wrapper function"""
    return fused_div_transpose_kernel_wrapper