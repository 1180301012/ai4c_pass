import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """
    Match full attention pattern including dropout:
    bmm -> softmax -> dropout -> bmm
    This pattern matches the complete attention computation with dropout
    """
    # First BMM: Q @ K^T
    attention_scores = torch.bmm(in_0, in_1)
    
    # Softmax on last dimension  
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    
    # Dropout operation (p=0.0 makes it a no-op)
    dropout_weights = torch.nn.functional.dropout(attention_weights, p=0.0, training=False)
    
    # Second BMM: dropout_weights @ V
    output = torch.bmm(dropout_weights, in_2)
    
    # Return all intermediates that need to be preserved
    # Note: The model returns only the final result, so we only need to return that
    return output

def replacement_args(in_0, in_1, in_2):
    """Extract arguments from matched nodes"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, 
    out_ptr,
    batch_size, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused attention kernel that computes Q @ K^T @ V in a single kernel
    """
    # Calculate thread position in grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute range each program will work on
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    offset = pid_b * batch_size * seq_len * seq_len
    
    # Create shared memory for Q, K accumulators
    q_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    k_acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    
    # Compute attention_scores = Q @ K^T
    for k in range(0, head_dim, BLOCK_K):
        k_end = min(k + BLOCK_K, head_dim)
        
        # Load Q block
        q_offset = (pid_b * seq_len + m_start) * head_dim + k
        q_block = tl.load(q_ptr + q_offset, 
                         mask=(m_start + tl.arange(BLOCK_M)) < seq_len and 
                              (k + tl.arange(BLOCK_K)) < head_dim,
                         other=0.0)
        
        # Load K block (transposed)
        k_offset = (pid_b * seq_len + n_start) * head_dim + k
        k_block = tl.load(k_ptr + k_offset,
                         mask=(n_start + tl.arange(BLOCK_N)) < seq_len and 
                              (k + tl.arange(BLOCK_K)) < head_dim,
                         other=0.0)
        
        # Matrix multiplication: Q @ K^T
        q_acc += q_block[:, None] * k_block[None, :]
    
    # Apply softmax to attention scores
    max_val = tl.maximum(q_acc, q_acc)
    sum_exp = tl.sum(tl.exp(q_acc - max_val), axis=1)
    attention_weights = tl.exp(q_acc - max_val) / sum_exp[:, None]
    
    # Compute output = attention_weights @ V
    o_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, head_dim, BLOCK_K):
        k_end = min(k + BLOCK_K, head_dim)
        
        # Load V block
        v_offset = (pid_b * seq_len + n_start) * head_dim + k
        v_block = tl.load(v_ptr + v_offset,
                         mask=(n_start + tl.arange(BLOCK_N)) < seq_len and 
                              (k + tl.arange(BLOCK_K)) < head_dim,
                         other=0.0)
        
        # Matrix multiplication: attention_weights @ V
        o_acc += attention_weights * v_block
    
    # Store result
    out_offset = offset + (m_start * seq_len + n_start) * head_dim
    tl.store(out_ptr + out_offset, o_acc,
             mask=(m_start + tl.arange(BLOCK_M)) < seq_len and 
                  (n_start + tl.arange(BLOCK_N)) < seq_len)

@torch.fx.wrap  
def fused_attention(q, k, v):
    """
    Wrapper function for fused attention computation
    """
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    head_dim = q.shape[2]
    
    # Handle different head dimensions from different models
    # Determine block sizes based on head_dim for optimal performance
    if head_dim <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32
    elif head_dim <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
    
    # Calculate grid size
    m_blocks = (seq_len + BLOCK_M - 1) // BLOCK_M
    n_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N
    grid_size = (m_blocks, n_blocks, batch_size)
    
    # Allocate output tensor
    output = torch.empty((batch_size, seq_len, head_dim), dtype=q.dtype, device=q.device)
    
    # Launch kernel
    fused_attention_kernel[grid_size](
        q_ptr=q,
        k_ptr=k, 
        v_ptr=v,
        out_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return output

def replacement_func():
    """Return the fused attention function"""
    return fused_attention