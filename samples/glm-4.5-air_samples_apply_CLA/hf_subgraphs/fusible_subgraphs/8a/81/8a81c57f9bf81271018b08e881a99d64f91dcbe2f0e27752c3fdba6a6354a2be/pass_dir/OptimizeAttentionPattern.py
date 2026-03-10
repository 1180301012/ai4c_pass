import torch
import triton
import triton.language as tl

def pattern(query, key, value, attn_mask):
    # Scaled dot product attention
    attention_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    return attention_output

def replacement_args(query, key, value, attn_mask):
    return (query, key, value, attn_mask)

@triton.jit
def scaled_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    attn_mask_ptr,
    out_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid: (batch, head, m_block, n_block)
    batch = tl.program_id(0)
    head = tl.program_id(1)
    m_block = tl.program_id(2)
    n_block = tl.program_id(3)
    
    # Offsets within the block
    m_base = batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + m_block * BLOCK_SIZE_M
    n_base = batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + n_block * BLOCK_SIZE_N
    
    # Memory offsets for Q, K, V
    q_offsets = m_base * head_dim + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :]
    k_offsets = n_base * head_dim + tl.arange(0, BLOCK_SIZE_N)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :]
    v_offsets = n_base * head_dim + tl.arange(0, BLOCK_SIZE_N)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :]
    out_offsets = m_base * head_dim + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :]
    
    # Load Q, K, V with masking
    q = tl.load(q_ptr + q_offsets, mask=q_offsets < batch_size * num_heads * seq_len * head_dim, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=k_offsets < batch_size * num_heads * seq_len * head_dim, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=v_offsets < batch_size * num_heads * seq_len * head_dim, other=0.0)
    
    # Compute Q * K^T
    qk = tl.dot(q, k, trans_b=True) * scale
    
    # Apply attention mask if provided
    if attn_mask_ptr is not None:
        mask_val = tl.load(attn_mask_ptr + q_offsets // head_dim)
        # Use Triton operations instead of torch.where
        mask = mask_val > 0
        qk = tl.where(mask, qk, float('-inf'))
    
    # Softmax
    max_val = tl.maximum(tl.max(qk, axis=1), tl.zeros_like(max_val := tl.zeros(1, dtype=tl.float32)))
    exp_qk = tl.exp(qk - max_val)
    sum_exp = tl.sum(exp_qk, axis=1, keepdim=True)
    exp_qk = exp_qk / sum_exp
    
    # Compute attention output
    attention = tl.dot(exp_qk, v)
    
    # Store result
    tl.store(out_ptr + out_offsets, attention)

@torch.fx.wrap
def optimized_attention(query, key, value, attn_mask):
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    scale = 1.0 / (head_dim ** 0.5)
    
    # Create output tensor
    out = torch.empty_like(query)
    
    # Block sizes for optimal GPU utilization
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Calculate grid size
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    scaled_attention_kernel[(batch_size, num_heads, num_blocks_m, num_blocks_n)](
        q_ptr=query,
        k_ptr=key,
        v_ptr=value,
        attn_mask_ptr=attn_mask if attn_mask is not None else None,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        scale=scale,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )
    
    return out

def replacement_func():
    return optimized_attention