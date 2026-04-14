import torch
import triton
import triton.language as tl

def pattern(in_3):
    """Optimize in_3.mean(-2) operation"""
    tmp_3 = in_3.mean(-2)
    return tmp_3

def replacement_args(in_3):
    """Extract arguments for mean operation optimization"""
    return (in_3,)

@triton.jit
def mean_kernel(
    x_ptr,              # input tensor [batch, seq_len, features]
    out_ptr,            # output tensor [batch, features]
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    """High-performance mean operation along dimension -2 (seq_len dimension)"""
    # Calculate program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    seq_offsets = tl.arange(0, BLOCK_SIZE_SEQ)
    
    # Create masks
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < features
    seq_mask = seq_offsets < seq_len
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop through sequence dimension in blocks
    for k in range(0, seq_len, BLOCK_SIZE_SEQ):
        # Calculate current sequence block offsets
        current_seq_offsets = k + seq_offsets
        current_seq_mask = current_seq_offsets < seq_len
        
        # Load elements for current sequence block
        base_ptr = x_ptr + m_offsets[:, None, None] * (seq_len * features) + \
                   current_seq_offsets[None, None, :] * features + \
                   n_offsets[None, :, None]
        
        # Load input block
        values = tl.load(base_ptr, 
                         mask=m_mask[:, None, None] & n_mask[None, :, None] & current_seq_mask[None, None, :], 
                         other=0.0)
        
        # Sum along sequence dimension (current block)
        summed_values = tl.sum(values, axis=2)
        accumulator += summed_values
    
    # Divide by total sequence length to get mean
    mean_values = accumulator / seq_len
    
    # Store result
    out_base = out_ptr + m_offsets[:, None] * features + n_offsets[None, :]
    tl.store(out_base, mean_values, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def triton_mean(x):
    """High-performance mean operation along dimension -2 using Triton"""
    batch_size, seq_len, features = x.shape
    
    # Set block sizes based on available GPU resources
    BLOCK_SIZE_M = 64   # Batch block size
    BLOCK_SIZE_N = 32   # Features block size
    BLOCK_SIZE_SEQ = 32  # Sequence dimension block size
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output
    out = torch.empty((batch_size, features), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    mean_kernel[(grid_m, grid_n)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return out

def replacement_func():
    """Returns the optimized mean function"""
    return triton_mean