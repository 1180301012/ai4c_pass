import torch
import triton
import triton.language as tl

@triton.jit
def attention_transpose_reshape_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    attn_mask_ptr,
    out_ptr,
    batch_size,
    n_heads,
    seq_len,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs - 3D grid for batch, heads, and output blocks
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_id = tl.program_id(2)
    
    # Compute offsets for this program
    m_offset = m_id * BLOCK_M
    n_offset = tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)
    
    # Create masks
    m_mask = m_offset < seq_len
    n_mask = n_offset < seq_len
    k_mask = k_offset < head_dim
    
    # Initialize output accumulator
    accumulator = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
    
    # MainAttention computation: Q @ K^T @ V
    for k in range(0, seq_len, BLOCK_K):
        k_start = k + k_offset
        
        # Load query block: [BLOCK_M, head_dim]
        q_ptrs = query_ptr + batch_id * n_heads * seq_len * head_dim + head_id * seq_len * head_dim + (m_offset + tl.arange(0, BLOCK_M)) * head_dim + k_start
        q = tl.load(q_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load key block: [seq_len, head_dim]
        k_ptrs_base = key_ptr + batch_id * n_heads * seq_len * head_dim + head_id * seq_len * head_dim + k_start
        k = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for i in range(BLOCK_M):
            if m_offset + i < seq_len:
                k_ptrs = k_ptrs_base + (m_offset + i) * head_dim
                k[i, :] = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Compute attention scores
        scores = tl.dot(q, k, out_type=tl.float32) / tl.sqrt(head_dim)
        
        # Apply attention mask if provided
        if attn_mask_ptr is not None:
            mask_ptrs = attn_mask_ptr + batch_id * seq_len * seq_len + (m_offset + tl.arange(0, BLOCK_M)) * seq_len + k_start
            mask = tl.load(mask_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            scores = scores * mask - 1e9 * (1 - mask)
        
        # Convert to attention weights
        weights = tl.softmax(scores, dim=1)
        
        # Load value block and compute weighted sum
        v_ptrs_base = value_ptr + batch_id * seq_len * head_dim + (m_offset + tl.arange(0, BLOCK_M)) * head_dim
        v = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for i in range(BLOCK_M):
            if m_offset + i < seq_len:
                v_ptrs = v_ptrs_base + (m_offset + i) * head_dim
                v[i, :] = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        # Compute weighted sum
        weighted_v = tl.dot(weights, v, out_type=tl.float32)
        accumulator += weighted_v
    
    # Store result with implicit transpose and reshape
    # The attention result naturally goes to [batch, n_heads, seq_len, head_dim] format
    seq_len_offset = tl.arange(0, BLOCK_M) if BLOCK_M > 1 else tl.arange(0, 1)
    out_base = out_ptr + batch_id * n_heads * seq_len * head_dim + head_id * seq_len * head_dim + (m_offset + seq_len_offset) * head_dim
    tl.store(out_base, accumulator, mask=m_mask[:, None])

@torch.fx.wrap
def attention_transpose_reshape_fusion(query, key, value, attn_mask, reshape_shape):
    batch_size, seq_len, n_heads, head_dim = query.shape
    
    # Output shape will be [batch_size, *reshape_shape]
    output = torch.empty((batch_size, *reshape_shape), dtype=query.dtype, device=query.device)
    
    # Set block sizes
    BLOCK_M = 64  # sequence length dimension
    BLOCK_N = 128  # key sequence dimension  
    BLOCK_K = 32   # head dimension
    
    # Calculate grid sizes
    grid_batch = batch_size
    grid_heads = n_heads  
    grid_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    
    # Launch kernel
    attention_transpose_reshape_kernel[(grid_batch, grid_heads, grid_m)](
        query_ptr=query,
        key_ptr=key,
        value_ptr=value,
        attn_mask_ptr=attn_mask,
        out_ptr=output.reshape(batch_size, seq_len, head_dim),  # Flatten for easier pointer arithmetic
        batch_size=batch_size,
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output

def pattern(query, key, value, attn_mask):
    tmp_5 = torch.nn.functional.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    tmp_6 = tmp_5.transpose(1, 2)
    return tmp_6

def replacement_args(query, key, value, attn_mask):
    # Simple output shape without conditional logic
    # Use a standard reshape pattern that works for most transformer models
    batch_size, n_heads, seq_len, head_dim = query.shape
    # Common output pattern: [batch_size, seq_len, hidden_dim] where hidden_dim = n_heads * head_dim
    hidden_dim = n_heads * head_dim
    return (query, key, value, attn_mask, (batch_size, seq_len, hidden_dim))

def replacement_func():
    return attention_transpose_reshape_fusion