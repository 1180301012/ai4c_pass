import torch
import triton
import triton.language as tl

def pattern(tensor):
    """Pattern matches the transpose operation for attention K matrix"""
    return tensor.transpose(-2, -1)

@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_heads,
    seq_len,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized Triton kernel for matrix transpose in attention"""
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # Compute input coordinates
    input_batch_offset = pid_batch * n_heads * seq_len * head_dim
    input_head_offset = pid_head * seq_len * head_dim
    input_m_offset = pid_m * BLOCK_SIZE_M
    input_n_offset = pid_n * BLOCK_SIZE_N
    
    # Compute output coordinates (transposed)
    output_batch_offset = pid_batch * n_heads * head_dim * seq_len  # head_dim and seq_len swapped
    output_head_offset = pid_head * head_dim * seq_len
    output_m_offset = pid_n * BLOCK_SIZE_N  # m and n swapped
    output_n_offset = pid_m * BLOCK_SIZE_M  # m and n swapped
    
    # Load input block
    m_offsets = input_m_offset + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = input_n_offset + tl.arange(0, BLOCK_SIZE_N)
    
    mask = (m_offsets[:, None] < seq_len) & (n_offsets[None, :] < head_dim)
    
    input_base_ptr = input_ptr + (
        input_batch_offset + 
        input_head_offset + 
        m_offsets[:, None] * head_dim + 
        n_offsets[None, :]
    )
    
    input_vals = tl.load(input_base_ptr, mask=mask)
    
    # Store output block (transposed)
    output_base_ptr = output_ptr + (
        output_batch_offset + 
        output_head_offset + 
        output_m_offset * head_dim +  # Note: using head_dim for stride here
        output_n_offset[None, :] +   # m and n swapped
        tl.arange(0, BLOCK_SIZE_N)[None, :] * head_dim  # stride by head_dim
    )
    
    # Transpose the data
    output_vals = input_vals.T
    
    tl.store(output_base_ptr, output_vals, mask=mask)

@torch.fx.wrap
def optimized_attention_transpose(tensor):
    """Optimized transpose for attention K matrix"""
    batch_size = tensor.shape[0]
    n_heads = tensor.shape[1]
    seq_len = tensor.shape[2]
    head_dim = tensor.shape[3]
    
    # Use small block sizes for better memory locality
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    # Launch grid dimensions
    grid_batch = batch_size
    grid_heads = n_heads
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output tensor
    output = torch.empty(batch_size, n_heads, head_dim, seq_len,
                        dtype=tensor.dtype, device=tensor.device)
    
    # Launch kernel
    transpose_kernel[(grid_batch, grid_heads, grid_m, grid_n)](
        input_ptr=tensor,
        output_ptr=output,
        n_batch=batch_size,
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_args(tensor):
    """Extract arguments for the transpose kernel"""
    return (tensor,)

def replacement_func():
    """Return the optimized transpose function"""
    return optimized_attention_transpose