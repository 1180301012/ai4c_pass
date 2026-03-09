import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, tmp_4, in_2):
    # Scaled dot product attention
    tmp_5 = torch.nn.functional.scaled_dot_product_attention(query=in_5, key=in_4, value=tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    return tmp_5

def replacement_args(in_5, in_4, tmp_4, in_2):
    return (in_5, in_4, tmp_4, in_2)

@triton.jit
def optimized_attention_kernel(
    query_ptr, key_ptr, value_ptr, attn_mask_ptr, output_ptr,
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    BLOCK_SIZE_M: tl.constexpr,  # seq_len dimension
    BLOCK_SIZE_N: tl.constexpr,  # seq_len_k dimension  
    BLOCK_SIZE_K: tl.constexpr,  # head_dim dimension
):
    # Each program handles one head in one batch
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Stride through batch and heads
    q_ptr = query_ptr + batch_id * num_heads * seq_len_q * head_dim + head_id * seq_len_q * head_dim
    k_ptr = key_ptr + batch_id * num_heads * seq_len_k * head_dim + head_id * seq_len_k * head_dim
    v_ptr = value_ptr + batch_id * num_heads * seq_len_k * head_dim + head_id * seq_len_k * head_dim
    o_ptr = output_ptr + batch_id * num_heads * seq_len_q * head_dim + head_id * seq_len_q * head_dim
    
    # Row offsets for query
    m_offset = tl.program_id(2) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Column offsets for key/value
    n_offset = tl.arange(0, BLOCK_SIZE_N)
    # Element offsets within head dimension
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    q_mask = m_offset < seq_len_q
    k_mask = n_offset < seq_len_k
    kv_mask = k_offset < head_dim
    
    # Load query vectors (one row at a time)
    q = tl.load(q_ptr + (m_offset[:, None] * head_dim + k_offset[None, :]), 
                mask=q_mask[:, None] & kv_mask[None, :], other=0.0)
    q = q.to(tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Compute attention
    for k_base in range(0, seq_len_k, BLOCK_SIZE_N):
        k_end = min(k_base + BLOCK_SIZE_N, seq_len_k)
        
        # Load key block
        k = tl.load(k_ptr + (n_offset[None, :] * head_dim + k_offset[:, None]), 
                    mask=k_mask[None, :] & kv_mask[:, None], other=0.0)
        k = k.to(tl.float32)
        k = k.T  # Transpose to match query dimensions
        
        # Compute Q*K^T / sqrt(head_dim)
        qk = tl.dot(q, k, out_dtype=tl.float32) * float(1.0 / (head_dim ** 0.5))
        
        # Apply attention mask if provided
        if attn_mask_ptr is not None:
            mask = tl.load(attn_mask_ptr + (m_offset[:, None] * seq_len_k + k_base + n_offset[None, :]), 
                          mask=q_mask[:, None] & k_mask[None, :], other=float('-inf'))
            qk = qk + mask
        
        # Softmax
        max_val = tl.maximum(tl.max(qk, axis=1), tl.full((BLOCK_SIZE_M, 1), float('-inf')))
        exp_qk = tl.exp(qk - max_val)
        sum_exp = tl.sum(exp_qk, axis=1)
        softmax_qk = exp_qk / sum_exp[:, None]
        
        # Load value block
        v = tl.load(v_ptr + ((k_base + n_offset)[:, None] * head_dim + k_offset[None, :]), 
                    mask=k_mask[:, None] & kv_mask[None, :], other=0.0)
        v = v.to(tl.float32)
        
        # Compute weighted sum
        acc = acc + tl.dot(softmax_qk, v, out_dtype=tl.float32)
    
    # Apply scaling factor and store result
    acc = acc.to(tl.float32 if query.dtype == torch.float32 else tl.float16)
    tl.store(o_ptr + (m_offset[:, None] * head_dim + k_offset[None, :]), acc, mask=q_mask[:, None] & kv_mask[None, :])

@torch.fx.wrap
def optimized_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    
    # For now assume key and value have same sequence length as query
    seq_len_k = key.shape[-2]
    
    # Create output tensor
    output = torch.empty_like(query)
    
    # Set up block sizes
    BLOCK_SIZE_M = 32  # Query sequence length block
    BLOCK_SIZE_N = 32  # Key sequence length block
    BLOCK_SIZE_K = min(32, head_dim)  # Head dimension block
    
    # Calculate grid dimensions
    grid_m = (seq_len_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len_k + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (batch_size, num_heads, grid_m)
    
    # Launch kernel
    optimized_attention_kernel[grid](
        query_ptr=query,
        key_ptr=key,
        value_ptr=value,
        attn_mask_ptr=attn_mask,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_scaled_dot_product_attention