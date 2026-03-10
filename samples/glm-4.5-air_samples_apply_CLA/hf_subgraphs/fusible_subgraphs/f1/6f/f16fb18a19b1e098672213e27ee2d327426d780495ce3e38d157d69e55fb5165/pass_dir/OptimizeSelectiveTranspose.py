import torch
import triton
import triton.language as tl

# Pattern matching for selective transpose
def pattern(x):
    # Transpose operation on the last two dimensions
    tmp_8 = x.transpose(-2, -1)
    return tmp_8

# Extract arguments for replacement
def replacement_args(x):
    return (x,)

# Optimized kernel for selective transpose
@triton.jit
def selective_transpose_kernel(
    x_ptr, out_ptr,
    seq_len, inner_dim, depth,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program identifiers for parallel execution
    pid_m = tl.program_id(0)  # sequence dimension (0-196)
    pid_n = tl.program_id(1)  # inner dimension
    
    # Block ranges within the program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Compute masks
    m_mask = m_offsets < seq_len
    n_mask = n_offsets < inner_dim
    
    # Create broadcastable masks for 2D operation
    mask = m_offsets[:, None] < seq_len & n_offsets[None, :] < inner_dim
    
    # Load input tensor in transposed order
    # Original shape: [1, inner_dim, seq_len, depth]
    # We want to transpose to: [1, seq_len, inner_dim, depth]
    # Load from (m, n) -> store to (n, m)
    x = tl.load(x_ptr + m_offsets[None, :] * inner_dim * depth + n_offsets[:, None] * depth,
               mask=mask, other=0.0)
    
    # Transpose the data - swap the two dimensions
    x_transposed = x.permute(1, 0)
    
    # Store result in the transposed order
    # Original offset formula for [1, inner_dim, seq_len, depth]:
    # offset = m * inner_dim * depth + n * depth + d
    # For transposed [1, seq_len, inner_dim, depth]:
    # offset = n * seq_len * depth + m * depth + d  (note: m and n are swapped)
    out_offset = n_offsets[:, None] * seq_len * depth + m_offsets[None, :] * depth
    
    # Store the transposed result
    tl.store(out_ptr + out_offset, x_transposed, mask=mask)

@torch.fx.wrap 
def selective_transpose_optimized(x):
    # Get tensor shape
    batch_size = x.shape[0]
    seq_len = x.shape[2]  # sequence dimension is at index 2
    inner_dim = x.shape[1]  # inner dimension is at index 1  
    depth = x.shape[3]    # depth dimension is at index 3
    
    # Create output tensor with transposed dimensions: [1, seq_len, inner_dim, depth]
    out_shape = (batch_size, seq_len, inner_dim, depth)
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Set block sizes for optimal GPU utilization
    BLOCK_SIZE_M = 64   # Process multiple sequence elements together
    BLOCK_SIZE_N = 64   # Process multiple inner features together
    
    # Calculate grid dimensions
    m_dim = seq_len
    n_dim = inner_dim
    
    # Launch kernel
    selective_transpose_kernel[(triton.cdiv(m_dim, BLOCK_SIZE_M), triton.cdiv(n_dim, BLOCK_SIZE_N))](
        x_ptr=x,
        out_ptr=out,
        seq_len=seq_len,
        inner_dim=inner_dim,
        depth=depth,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return selective_transpose_optimized