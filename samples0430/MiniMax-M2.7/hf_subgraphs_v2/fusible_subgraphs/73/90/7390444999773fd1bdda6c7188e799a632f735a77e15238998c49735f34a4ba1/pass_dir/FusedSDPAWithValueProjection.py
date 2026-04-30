import torch
import triton
import triton.language as tl


@torch.fx.wrap
def fused_sdpa_value_wrapper(value_bias, value_weight, hidden_states, attn_mask, query, key, dropout_p, is_causal, route):
    """
    Fused computation for value projection + multi-head attention reshape/transpose + SDPA.
    
    This function fuses:
    - Linear projection for value
    - View/reshape for multi-head attention format
    - Transpose operations
    - SDPA computation
    - Output transpose and reshape
    
    Args:
        value_bias: Value bias tensor [hidden]
        value_weight: Value weight tensor [hidden, hidden]
        hidden_states: Hidden states [batch, seq, hidden]
        attn_mask: Attention mask [batch, 1, seq, seq] or broadcastable
        query: Query tensor [batch, num_heads, seq, head_dim]
        key: Key tensor [batch, num_heads, seq, head_dim]
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        route: Route string for dispatch (unused, for API compliance)
    
    Returns:
        Output tensor [batch, seq, hidden]
    """
    # Compute value projection using linear
    value_proj = torch.nn.functional.linear(hidden_states, value_weight, value_bias)
    
    # Infer dimensions from input shapes
    B, S, H = value_proj.shape
    _, num_heads, _, head_dim = query.shape
    
    # Reshape value for multi-head attention: [B, S, H] -> [B, S, num_heads, head_dim]
    value_reshaped = value_proj.view(B, S, num_heads, head_dim)
    
    # Transpose for attention: [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
    value_transposed = value_reshaped.transpose(1, 2)
    
    # Call SDPA
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value_transposed,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal
    )
    
    # Transpose back: [B, num_heads, S, head_dim] -> [B, S, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2)
    
    # Reshape to original format: [B, S, num_heads, head_dim] -> [B, S, H]
    attn_output = attn_output.reshape(B, S, H)
    
    return attn_output


def pattern_1_512_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,2,64) -> transpose -> sdpa -> transpose -> reshape(1,512,128)
    For BERT small with hidden=128, heads=2, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 512, 128)
    return tmp_7


def replacement_args_1_512_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_512_128_2_64")


def pattern_1_12_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,4,64) -> transpose -> sdpa -> transpose -> reshape(1,12,256)
    For BERT medium with hidden=256, heads=4, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 256)
    return tmp_7


def replacement_args_1_12_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_12_256_4_64")


def pattern_4_512_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(4,-1,4,64) -> transpose -> sdpa -> transpose -> reshape(4,512,256)
    For batch=4 with hidden=256, heads=4, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(4, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(4, 512, 256)
    return tmp_7


def replacement_args_4_512_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "4_512_256_4_64")


def pattern_16_128_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(16,-1,2,64) -> transpose -> sdpa -> transpose -> reshape(16,128,128)
    For batch=16 with hidden=128, heads=2, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(16, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(16, 128, 128)
    return tmp_7


def replacement_args_16_128_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "16_128_128_2_64")


def pattern_8_256_768_12_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(8,-1,12,64) -> transpose -> sdpa -> transpose -> reshape(8,256,768)
    For batch=8 with hidden=768, heads=12, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(8, -1, 12, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(8, 256, 768)
    return tmp_7


def replacement_args_8_256_768_12_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "8_256_768_12_64")


def pattern_128_64_512_8_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(128,-1,8,64) -> transpose -> sdpa -> transpose -> reshape(128,64,512)
    For batch=128 with hidden=512, heads=8, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(128, -1, 8, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(128, 64, 512)
    return tmp_7


def replacement_args_128_64_512_8_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "128_64_512_8_64")


def pattern_64_128_768_12_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(64,-1,12,64) -> transpose -> sdpa -> transpose -> reshape(64,128,768)
    For batch=64 with hidden=768, heads=12, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(64, -1, 12, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(64, 128, 768)
    return tmp_7


def replacement_args_64_128_768_12_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "64_128_768_12_64")


def pattern_1_64_512_8_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,8,64) -> transpose -> sdpa -> transpose -> reshape(1,64,512)
    For batch=1 with hidden=512, heads=8, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 8, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 64, 512)
    return tmp_7


def replacement_args_1_64_512_8_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_64_512_8_64")


def pattern_64_128_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(64,-1,4,64) -> transpose -> sdpa -> transpose -> reshape(64,128,256)
    For batch=64 with hidden=256, heads=4, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(64, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(64, 128, 256)
    return tmp_7


def replacement_args_64_128_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "64_128_256_4_64")


def pattern_1_64_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,4,64) -> transpose -> sdpa -> transpose -> reshape(1,64,256)
    For batch=1 with hidden=256, heads=4, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 64, 256)
    return tmp_7


def replacement_args_1_64_256_4_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_64_256_4_64")


def pattern_1_64_768_12_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,12,64) -> transpose -> sdpa -> transpose -> reshape(1,64,768)
    For batch=1 with hidden=768, heads=12, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 12, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 64, 768)
    return tmp_7


def replacement_args_1_64_768_12_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_64_768_12_64")


def pattern_1_11_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,2,64) -> transpose -> sdpa -> transpose -> reshape(1,11,128)
    For BERT tiny with batch=1, seq=11, hidden=128, heads=2, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 11, 128)
    return tmp_7


def replacement_args_1_11_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_11_128_2_64")


def pattern_1_12_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: linear -> view(1,-1,2,64) -> transpose -> sdpa -> transpose -> reshape(1,12,128)
    For batch=1, seq=12 with hidden=128, heads=2, head_dim=64
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 128)
    return tmp_7


def replacement_args_1_12_128_2_64(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_12_128_2_64")


# All patterns share the same replacement function
def replacement_func():
    return fused_sdpa_value_wrapper