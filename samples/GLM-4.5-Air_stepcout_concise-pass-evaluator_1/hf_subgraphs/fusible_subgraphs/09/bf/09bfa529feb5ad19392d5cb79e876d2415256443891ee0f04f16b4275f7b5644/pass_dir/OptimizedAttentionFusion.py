import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Match the exact computation pattern from the graphs
    # Handle both dropout variants by testing the more common case first
    tmp_0 = x + y
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    tmp_3 = tmp_2.to(torch.float32)
    return (tmp_3,)

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_attention_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size, num_heads, seq_len_k, seq_len_q,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Calculate program IDs for 3D grid (batch, head, m_dim)
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute memory addresses for this program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Create broadcastable offsets for 4D tensor layout [batch, head, q_seq, k_seq]
    m_offsets = m_offsets[:, None]
    n_offsets = n_offsets[None, :]
    
    # Calculate total offsets with proper 4D layout
    # Each head has seq_len_q * seq_len_k elements
    head_offset = pid_h * seq_len_q * seq_len_k
    batch_offset = pid_b * num_heads * seq_len_q * seq_len_k
    offsets = batch_offset + head_offset + m_offsets * seq_len_k + n_offsets
    
    # Create bounds mask
    mask = (m_offsets < seq_len_q) & (n_offsets < seq_len_k)
    mask = mask.to(tl.int32)
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Element-wise addition
    sum_val = x + y
    
    # Softmax along the last dimension (k_seq dimension)
    max_val = tl.max(sum_val, axis=1, keepdim=True)
    exp_val = tl.exp(sum_val - max_val)
    sum_exp = tl.sum(exp_val, axis=1, keepdim=True)
    softmax_val = exp_val / (sum_exp + 1e-8)  # Add small epsilon for stability
    
    # Dropout with 0.0 is identity operation, so we just pass through
    # Store the result (already float32 from softmax)
    tl.store(out_ptr + offsets, softmax_val, mask=mask)

@torch.fx.wrap
def optimized_attention_forward(x, y):
    # Get 4D tensor shapes: [batch_size, num_heads, seq_len_q, seq_len_k]
    batch_size, num_heads, seq_len_q, seq_len_k = x.shape
    
    # Output tensor (already float32)
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Choose block sizes based on sequence length for good occupancy
    if seq_len_q <= 64:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
    elif seq_len_q <= 128:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
    elif seq_len_q <= 256:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
    
    # Calculate grid dimensions for 3D grid [batch, head, q_seq_blocks]
    num_blocks_m = (seq_len_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (num_blocks_m, num_heads, batch_size)
    
    # Launch the kernel
    optimized_attention_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_k=seq_len_k,
        seq_len_q=seq_len_q,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_attention_forward