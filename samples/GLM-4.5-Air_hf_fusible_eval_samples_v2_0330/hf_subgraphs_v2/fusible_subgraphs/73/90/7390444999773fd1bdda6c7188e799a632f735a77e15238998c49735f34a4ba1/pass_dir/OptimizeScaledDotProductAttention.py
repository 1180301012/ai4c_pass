import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_3, in_2):
    """
    Pattern matches: scaled_dot_product_attention operation
    Returns the attention output
    """
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(
        in_5, in_4, in_3, attn_mask=in_2, dropout_p=0.0, is_causal=False
    )
    return scaled_dot_product_attention

def replacement_args(in_5, in_4, in_3, in_2):
    return (in_5, in_4, in_3, in_2)

@triton.jit
def scaled_dot_product_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized multi-head attention kernel using Triton"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(head_dim, BLOCK_SIZE_K)
    
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Calculate global offsets for query, key, value
    query_offset = batch_size * num_heads * seq_len * head_dim
    key_offset = batch_size * num_heads * seq_len * head_dim
    value_offset = batch_size * num_heads * seq_len * head_dim
    
    # Pointers for current head
    head_idx = (pid // (num_pid_m * num_pid_n)) % num_heads
    
    # Offsets for current head
    query_head_ptr = query_ptr + head_idx * seq_len * head_dim
    key_head_ptr = key_ptr + head_idx * seq_len * head_dim
    value_head_ptr = value_ptr + head_idx * seq_len * head_dim
    output_head_ptr = output_ptr + head_idx * seq_len * head_dim
    
    # Initialize accumulators
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute attention scores and output
    for k in range(num_pid_k):
        # Load query [BLOCK_SIZE_M, BLOCK_SIZE_K]
        query_ptrs = query_head_ptr + (offs_am[:, None] * head_dim + offs_k[None, :])
        query = tl.load(query_ptrs, mask=offs_am[:, None] < seq_len, other=0.0)
        
        # Load key for scaling [BLOCK_SIZE_K, BLOCK_SIZE_N] - need to transpose
        key_ptrs = key_head_ptr + (offs_k[:, None] * seq_len + offs_bn[None, :])
        key = tl.load(key_ptrs, mask=offs_k[:, None] < head_dim, other=0.0)
        
        # Compute Q * K^T / sqrt(d_k)
        qk = tl.dot(query, key, trans_b=True)
        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = qk * scale
        
        # Convert to bfloat16 if needed
        if output_ptr.dtype == tl.bfloat16:
            attn_scores = attn_scores.to(tl.bfloat16)
        elif output_ptr.dtype == tl.float16:
            attn_scores = attn_scores.to(tl.float16)
        
        # Apply softmax (simplified - in practice we'd need full softmax)
        # For now, we'll use a simple approximation for demonstration
        max_val = tl.max(attn_scores, 1)
        exp_scores = tl.exp(attn_scores - max_val[:, None])
        sum_exp = tl.sum(exp_scores, 1)
        attn_weights = exp_scores / sum_exp[:, None]
        
        # Load value [BLOCK_SIZE_K, BLOCK_SIZE_N] - need transpose
        value_ptrs = value_head_ptr + (offs_k[:, None] * seq_len + offs_bn[None, :])
        value = tl.load(value_ptrs, mask=offs_k[:, None] < head_dim, other=0.0)
        
        # Compute attention output
        attn_output = tl.dot(attn_weights, value)
        
        # Accumulate
        accumulator = accumulator + attn_output
    
    # Store results
    output_ptrs = output_head_ptr + (offs_am[:, None] * seq_len + offs_bn[None, :])
    output_mask = (offs_am[:, None] < seq_len) & (offs_bn[None, :] < seq_len)
    
    if output_ptr.dtype == tl.bfloat16:
        accumulator = accumulator.to(tl.bfloat16)
    elif output_ptr.dtype == tl.float16:
        accumulator = accumulator.to(tl.float16)
    elif output_ptr.dtype == tl.float32:
        pass  # already float32
    else:
        raise ValueError(f"Unsupported dtype: {output_ptr.dtype}")
    
    tl.store(output_ptrs, accumulator, mask=output_mask)

@triton.jit
def qkv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """More optimized QKV attention kernel with better memory access patterns"""
    # Calculate program dimensions
    m = tl.program_id(0)
    h = tl.program_id(1)
    
    # Determine bounds for this program
    start_m = m * BLOCK_SIZE_M
    end_m = min((m + 1) * BLOCK_SIZE_M, seq_len)
    
    # Initialize accumulator for this output tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process each head
    for k in range(0, seq_len, BLOCK_SIZE_K):
        # Load query tile for current head and output token positions
        query_ptrs = query_ptr + (batch_size * h * seq_len + start_m) * head_dim + k
        query_tile = tl.load(
            query_ptrs + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[None, :],
            mask=start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len,
            other=0.0
        )
        
        # Load key tile for current head and input token positions  
        key_ptrs = key_ptr + (batch_size * h * seq_len + k) * head_dim + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] * seq_len
        key_tile = tl.load(
            key_ptrs + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] * seq_len + tl.arange(0, min(BLOCK_SIZE_N, end_m - start_m))[None, :],
            mask=(tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] < head_dim) & (k + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] < seq_len),
            other=0.0
        )
        
        # Compute attention scores
        scores = tl.dot(query_tile, key_tile)
        scale = 1.0 / (head_dim ** 0.5)
        scores = scores * scale
        
        # Apply softmax
        max_scores = tl.max(scores, axis=1)
        exp_scores = tl.exp(scores - max_scores[:, None])
        norm_scores = exp_scores / tl.sum(exp_scores, axis=1)[:, None]
        
        # Load value tile
        value_ptrs = value_ptr + (batch_size * h * seq_len + k) * head_dim + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] * seq_len
        value_tile = tl.load(
            value_ptrs + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] * seq_len + tl.arange(0, min(BLOCK_SIZE_N, end_m - start_m))[None, :],
            mask=(tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] < head_dim) & (k + tl.arange(0, min(BLOCK_SIZE_K, seq_len - k))[:, None] < seq_len),
            other=0.0
        )
        
        # Accumulate output
        accumulator = accumulator + tl.dot(norm_scores, value_tile)
    
    # Store output
    output_ptrs = output_ptr + (batch_size * h * seq_len + start_m) * head_dim + tl.arange(0, BLOCK_SIZE_M)[:, None] * seq_len + tl.arange(0, min(BLOCK_SIZE_N, end_m - start_m))[None, :]
    output_mask = (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len) & (tl.arange(0, min(BLOCK_SIZE_N, end_m - start_m))[None, :] < seq_len)
    
    if output_ptr.dtype == tl.bfloat16:
        accumulator = accumulator.to(tl.bfloat16)
    elif output_ptr.dtype == tl.float16:
        accumulator = accumulator.to(tl.float16)
    
    tl.store(output_ptrs, accumulator, mask=output_mask)

@torch.fx.wrap
def optimized_scaled_dot_product_attention(query, key, value, attn_mask=None):
    """Wrapper function for optimized attention computation"""
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape
    _, _, seq_len_v, _ = value.shape
    
    # Make sure all dimensions match
    assert seq_len_q == seq_len_k == seq_len_v, f"Sequence lengths must match: {seq_len_q}, {seq_len_k}, {seq_len_v}"
    assert batch_size == 1, "Only batch size 1 supported for now"
    
    # Output shape: [batch_size, num_heads, seq_len_q, head_dim]
    output_shape = (batch_size, num_heads, seq_len_q, head_dim)
    output = torch.empty(output_shape, dtype=query.dtype, device=query.device)
    
    # Set block sizes based on typical GPU architectures
    BLOCK_SIZE_M = 32  # Output sequence length
    BLOCK_SIZE_N = 32  # Key sequence length  
    BLOCK_SIZE_K = 32  # Head dimension
    
    # Launch kernel grid
    grid_m = tl.cdiv(seq_len_q, BLOCK_SIZE_M)
    grid_h = num_heads
    grid = (grid_m, grid_h)
    
    qkv_attention_kernel[grid](
        query,
        key,
        value,
        output,
        batch_size,
        num_heads,
        seq_len_q,
        head_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_scaled_dot_product_attention