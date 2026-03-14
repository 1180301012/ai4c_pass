import torch
import triton
import triton.language as tl

@triton.jit
def full_attention_fusion_kernel(
    attn_scores_ptr,
    values_ptr,
    output_ptr,
    scale_factor,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Full attention fusion kernel: scale + softmax + matmul + transpose"""
    # Program IDs for parallel execution
    m_pid = tl.program_id(0)  # seq dimension
    n_pid = tl.program_id(1)  # head_dim dimension
    b_pid = tl.program_id(2)  # batch dimension
    
    # Calculate ranges
    m_offset = m_pid * BLOCK_M  # seq offset
    n_offset = n_pid * BLOCK_N  # head_dim offset
    
    # Shared memory tiles
    attn_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    values_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    output_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load attention scores tile with scaling
    for m in range(BLOCK_M):
        for k in range(num_heads):  # num_heads instead of BLOCK_K for simplicity
            m_idx = m_offset + m
            if m_idx < seq_len:
                attn_idx = m_idx * num_heads + k
                attn_val = tl.load(attn_scores_ptr + attn_idx)
                # Apply scaling immediately
                scaled_attn = attn_val * scale_factor
                attn_tile[m, k % BLOCK_K] = scaled_attn
    
    # Load values tile
    for k in range(num_heads):  # num_heads instead of BLOCK_K for simplicity
        for n in range(BLOCK_N):
            n_idx = n_offset + n
            if n_idx < head_dim:
                values_idx = k * head_dim + n_idx
                values_val = tl.load(values_ptr + values_idx)
                values_tile[k % BLOCK_K, n] = values_val
    
    # Apply softmax and perform attention computation
    for m in range(BLOCK_M):
        if m_offset + m < seq_len:
            # Compute max for softmax along the head dimension (k)
            max_val = -tl.float32('inf')
            for k in range(num_heads):
                attn_val = attn_tile[m, k % BLOCK_K]
                if attn_val > max_val:
                    max_val = attn_val
            
            # Compute sum of exp for softmax normalization
            sum_exp = 0.0
            for k in range(num_heads):
                attn_val = attn_tile[m, k % BLOCK_K]
                exp_val = tl.exp(attn_val - max_val)
                sum_exp += exp_val
            
            # Compute attention weights and perform weighted sum
            for n in range(BLOCK_N):
                acc = 0.0
                for k in range(num_heads):
                    attn_val = attn_tile[m, k % BLOCK_K]
                    values_val = values_tile[k % BLOCK_K, n]
                    # Compute softmax weight
                    softmax_val = tl.exp(attn_val - max_val) / sum_exp
                    # Weighted sum
                    acc += softmax_val * values_val
                
                # Store result in transposed layout
                m_idx = m_offset + m
                n_idx = n_offset + n
                if m_idx < seq_len and n_idx < head_dim:
                    # Transposed output: [head_dim, seq_len] storage
                    output_idx = n_idx * seq_len + m_idx
                    output_tile[m, n] = acc
    
    # Store output tile
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            m_idx = m_offset + m
            n_idx = n_offset + n
            if m_idx < seq_len and n_idx < head_dim:
                output_idx = n_idx * seq_len + m_idx  # Transposed storage
                tl.store(output_ptr + output_idx, output_tile[m, n])

@torch.fx.wrap
def full_attention_fusion(attn_scores, values, scale=0.0625):
    """Full fusion: scale + softmax + matmul + transpose in one kernel"""
    batch_size, seq_len, num_heads = attn_scores.shape
    values_batch, values_heads, head_dim = values.shape
    
    assert batch_size == values_batch
    assert num_heads == values_heads
    
    # Create output directly in transposed shape [batch, head_dim, seq_len]
    output_shape = (batch_size, head_dim, seq_len)
    output = torch.empty(output_shape, dtype=attn_scores.dtype, device=attn_scores.device)
    
    # Triton kernel blocking configuration
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 16  # Optimized for attention patterns
    
    # Process each batch in parallel
    for b in range(batch_size):
        # Get pointer to current batch data
        attn_scores_b = attn_scores[b, :, :].contiguous()
        values_b = values[b, :, :].contiguous()
        
        # Create batch output tile
        batch_output = torch.zeros((head_dim, seq_len), dtype=attn_scores.dtype, device=attn_scores.device)
        
        # Calculate grid dimensions
        m_dim = (seq_len + BLOCK_M - 1) // BLOCK_M
        n_dim = (head_dim + BLOCK_N - 1) // BLOCK_N
        
        # Launch Triton kernel
        full_attention_fusion_kernel[(m_dim, n_dim, 1)](
            attn_scores_ptr=attn_scores_b,
            values_ptr=values_b,
            output_ptr=batch_output,
            scale_factor=scale,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )
        
        # Store in final output
        output[b, :, :] = batch_output
    
    return output

def pattern(in_0, in_1):
    """Pattern: scale + softmax + matmul + transpose (full attention)"""
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized full attention fusion function"""
    return full_attention_fusion