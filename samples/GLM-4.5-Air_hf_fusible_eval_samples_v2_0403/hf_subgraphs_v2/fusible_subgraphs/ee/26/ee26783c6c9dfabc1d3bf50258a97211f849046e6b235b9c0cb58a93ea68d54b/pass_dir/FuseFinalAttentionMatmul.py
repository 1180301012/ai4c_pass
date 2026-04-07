import torch
import triton
import triton.language as tl

# Pattern matching for the final softmax + matmul sequence
def pattern(tmp_12, in_4):
    # Original final computation sequence:
    tmp_13 = tmp_12.softmax(dim = -1)
    matmul_1 = tmp_13 @ in_4
    output = matmul_1.transpose(-1, -2)
    
    return output


def replacement_args(tmp_12, in_4):
    return (tmp_12, in_4)


@triton.jit
def final_attention_kernel(
    attn_scores_ptr,
    v_ptr,
    output_ptr,
    batch_size, seq_len, heads,
    v_heads,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Optimized kernel that fuses softmax + matmul + transpose for attention
    """
    m_pid = tl.program_id(0)  # batch * heads
    k_pid = tl.program_id(1)  # seq_len (from v)
    n_pid = tl.program_id(2)  # seq_len (from attention scores), but we'll use block processing
    
    batch = m_pid // heads
    head_idx = m_pid % heads
    
    # Initialize accumulator for output
    acc = 0.0
    
    # Apply softmax to attention scores and multiply with V
    seq_len_k = seq_len  # Input sequence length (attention scores shape [seq_len, seq_len])
    seq_len_v = v_ptr.shape[1]  # Output sequence length (v shape [v_heads, ...])
    
    # For each position in the output sequence
    for k in range(seq_len_v):
        # Load attention scores for this head and batch position: [seq_len]
        attn_offset = (batch * heads * seq_len_k * seq_len_k + 
                      head_idx * seq_len_k * seq_len_k + 
                      n_pid * seq_len_k + k)
        attn_val = tl.load(attn_scores_ptr + attn_offset, mask=k < seq_len_k)
        
        # Apply max for numerical stability
        max_val = tl.max(attn_val)
        exp_val = tl.exp(attn_val - max_val)
        
        # Normalize
        sum_exp = tl.sum(exp_val)
        softmax_val = exp_val / sum_exp
        
        # Load corresponding value: [v_heads]
        v_offset = k * v_heads + n_pid
        v_val = tl.load(v_ptr + v_offset, mask=n_pid < v_heads)
        
        # Accumulate
        acc += tl.sum(softmax_val * v_val)
    
    # Store result with transpose applied
    output_offset = (batch * heads * seq_len + head_idx * seq_len + n_pid)
    tl.store(output_ptr + output_offset, acc)


@torch.fx.wrap
def fused_attention_final(attn_scores, v):
    """
    Fused function that replaces softmax + matmul + transpose
    """
    # Get input dimensions
    batch_size = attn_scores.shape[0]
    seq_len = attn_scores.shape[1]
    heads = attn_scores.shape[2]
    
    # For v tensor: [batch, seq_len, heads] or [batch, seq_len, v_heads]
    if len(v.shape) == 4:
        # Case 1: [batch, seq_len, heads, v_heads] (reshape to [batch, seq_len, v_heads*heads])
        v_heads = v.shape[3]
        v_reshaped = v.reshape(batch_size, seq_len, heads * v_heads)
    else:
        # Case 2: [batch, seq_len, heads] (v_heads = heads)
        v_heads = heads
        v_reshaped = v
    
    # Output shape: [batch, heads, seq_len, v_heads]
    output_shape = (batch_size, heads, seq_len, v_heads)
    output = torch.empty(output_shape, dtype=attn_scores.dtype, device=attn_scores.device)
    
    # For small cases, fallback to native implementation
    total_elements = batch_size * heads * seq_len * seq_len
    if total_elements < 100000:  # Threshold for kernel overhead
        # Fallback to native implementation
        attn_softmax = attn_scores.softmax(dim=-1)
        matmul_result = attn_softmax @ v_reshaped
        output = matmul_result.transpose(1, 2)
    else:
        # Launch Triton kernel for larger cases
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        # Grid: [batch*heads, seq_len, seq_len]
        grid = (batch_size * heads, seq_len, seq_len)
        
        final_attention_kernel[grid](
            attn_scores_ptr=attn_scores,
            v_ptr=v_reshaped,
            output_ptr=output,
            batch_size=batch_size,
            seq_len=seq_len,
            heads=heads,
            v_heads=v_heads,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K
        )
    
    return output


def replacement_func():
    return fused_attention_final