import torch
import triton
import triton.language as tl


# Pattern matching function - matches two consecutive dropout(0.0) operations
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the multi-head attention + double dropout pattern.
    Since dropout(x, 0.0, False, False) is identity, we can fuse these.
    """
    # Multi-head attention forward
    tmp_4 = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0, in_1, in_0,
        training=False, key_padding_mask=None, need_weights=True, attn_mask=None, 
        average_attn_weights=True, is_causal=False
    )
    # Extract attention output from tuple
    tmp_5 = tmp_4[0]
    # First dropout (no-op with p=0.0)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    # Second dropout (no-op with p=0.0)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments needed for the optimized kernel."""
    return (in_0, in_1, in_2, in_3, in_4)


# Triton kernel for multi-head attention with fused dropout elimination
@triton.jit
def mha_fused_kernel(
    query_ptr, key_ptr, value_ptr,
    in_proj_weight_ptr, in_proj_bias_ptr,
    out_proj_weight_ptr, out_proj_bias_ptr,
    out_ptr,
    n_elements,
    n_heads: tl.constexpr,
    embed_dim: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused multi-head attention kernel that eliminates the redundant dropout operations.
    """
    # Calculate indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Reshape for attention computation
    batch_size = 150
    seq_len = 1
    
    # Load query/key/value (they're the same - self-attention with in_4)
    q = tl.load(query_ptr + offsets * query_ptr.stride(0), mask=mask, other=0.0)
    k = tl.load(key_ptr + offsets * key_ptr.stride(0), mask=mask, other=0.0)
    v = tl.load(value_ptr + offsets * value_ptr.stride(0), mask=mask, other=0.0)
    
    # Load projection weights
    in_proj_w0 = tl.load(in_proj_weight_ptr + offsets * in_proj_weight_ptr.stride(0), mask=mask, other=0.0)
    
    # Output is the same as input after in-projection and out-projection with no dropout
    # This is essentially: out = out_proj(in_proj(x))
    result = q  # Placeholder - actual implementation needs full attention computation
    
    tl.store(out_ptr + offsets * out_ptr.stride(0), result, mask=mask)


@torch.fx.wrap
def mha_fused_wrapper(in_0, in_1, in_2, in_3, in_4):
    """
    Wrapper for the fused MHA kernel.
    Since we need to properly implement the attention, we'll use a simpler approach:
    compute the attention and directly return the result (eliminating dropouts).
    """
    # Input tensor shape: [150, 1, 512]
    # Multi-head attention with 8 heads, head_dim = 64
    
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads  # 64
    batch_size, seq_len, _ = in_4.shape
    
    # Compute Q, K, V through in_proj_weight [1536, 512] and in_proj_bias [1536]
    # in_proj splits into Q, K, V each of shape [1536] = 3 * 512
    
    # Reshape input for batch matrix multiplication
    # Shape: [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, embed_dim]
    x = in_4
    
    # Manual attention computation
    # Q = x @ W_q, K = x @ W_k, V = x @ W_v
    # Using in_proj_weight [1536, 512] and in_proj_bias [1536]
    
    # Split the in_proj weight and bias into Q, K, V parts
    w_q = in_3[:512, :]  # [512, 512]
    w_k = in_3[512:1024, :]  # [512, 512]
    w_v = in_3[1024:, :]  # [512, 512]
    
    b_q = in_2[:512]  # [512]
    b_k = in_2[512:1024]  # [512]
    b_v = in_2[1024:]  # [512]
    
    # Compute Q, K, V
    # Shape of x: [150, 1, 512]
    q = torch.matmul(x, w_q.t()) + b_q  # [150, 1, 512]
    k = torch.matmul(x, w_k.t()) + b_k  # [150, 1, 512]
    v = torch.matmul(x, w_v.t()) + b_v  # [150, 1, 512]
    
    # Reshape for multi-head attention: [B, N, D] -> [B, num_heads, N, head_dim]
    B, N, D = q.shape  # B=150, N=1, D=512
    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, 8, 1, 64]
    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, 8, 1, 64]
    v = v.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, 8, 1, 64]
    
    # Scaled dot-product attention
    scale = 1.0 / (head_dim ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, 8, 1, 1]
    attn = torch.softmax(attn, dim=-1)
    
    # Apply attention to values
    out = torch.matmul(attn, v)  # [B, 8, 1, 64]
    
    # Reshape back: [B, num_heads, N, head_dim] -> [B, N, D]
    out = out.transpose(1, 2).contiguous().view(B, N, D)  # [150, 1, 512]
    
    # Output projection
    out = torch.matmul(out, in_1.t()) + in_0  # [150, 1, 512]
    
    # Dropout is p=0.0 so we skip it (identity)
    # Second dropout is also skipped (identity)
    
    return (out,)


def replacement_func():
    """
    Return the optimized function that eliminates redundant dropouts.
    """
    return mha_fused_wrapper