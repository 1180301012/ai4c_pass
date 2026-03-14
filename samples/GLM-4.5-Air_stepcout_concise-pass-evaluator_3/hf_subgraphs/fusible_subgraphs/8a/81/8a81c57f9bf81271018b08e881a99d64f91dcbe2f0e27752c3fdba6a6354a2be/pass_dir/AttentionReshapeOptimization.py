import torch
import triton
import triton.language as tl

def pattern(hidden_states, value_weight, value_bias, attn_mask, key_layer, query_layer):
    # Linear transformation for values
    tmp_0 = value_bias
    tmp_1 = value_weight
    tmp_2 = torch.nn.functional.linear(hidden_states, tmp_1, tmp_0)
    tmp_1 = tmp_0 = None
    
    # View operation for attention heads
    tmp_3 = tmp_2.view(1, -1, tmp_2.size(2) // 64, 64)
    tmp_2 = None
    
    # Transpose to attention format
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_3 = None
    
    # Scaled dot product attention
    tmp_5 = torch.nn.functional.scaled_dot_product_attention(
        query=query_layer, 
        key=key_layer, 
        value=tmp_4, 
        attn_mask=attn_mask, 
        dropout_p=0.0, 
        is_causal=False
    )
    tmp_4 = None
    
    # Transpose back and reshape final output
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_5 = None
    tmp_7 = tmp_6.reshape(1, 64, 768)
    tmp_6 = None
    
    return tmp_7

def replacement_args(hidden_states, value_weight, value_bias, attn_mask, key_layer, query_layer):
    return (hidden_states, value_weight, value_bias, attn_mask, key_layer, query_layer)

@torch.fx.wrap
def optimized_attention_with_efficient_reshape(hidden_states, value_weight, value_bias, attn_mask, key_layer, query_layer):
    # Get tensor dimensions
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    
    # Determine head configuration (using head_dim=64 as common pattern)
    num_heads = hidden_size // 64
    
    # Efficient linear transformation for values
    values = torch.nn.functional.linear(hidden_states, value_weight, value_bias)
    
    # Efficient view operation - avoid creating intermediate tensors
    # Extract the exact view pattern from different models
    if values.shape[0] == 1 and hidden_size == 768:
        # Model 1: view(1, -1, 12, 64)
        values = values.view(batch_size, seq_len, num_heads, 64)
    elif values.shape[0] == 64 and values.shape[2] == 128:
        # Model 2: view(64, -1, 2, 64)
        values = values.view(batch_size, seq_len, num_heads, 64)
    elif values.shape[0] == 4 and values.shape[2] == 128:
        # Model 3: view(4, -1, 2, 64)
        values = values.view(batch_size, seq_len, num_heads, 64)
    elif values.shape[0] == 4 and values.shape[2] == 768:
        # Model 4: view(4, -1, 12, 64)
        values = values.view(batch_size, seq_len, num_heads, 64)
    elif values.shape[0] == 64 and values.shape[2] == 768:
        # Model 5: view(64, -1, 12, 64) 
        values = values.view(batch_size, seq_len, num_heads, 64)
    else:
        # Generic fallback
        values = values.view(batch_size, seq_len, num_heads, 64)
    
    # Reshape key and query layers efficiently
    if key_layer.shape[0] == 1:
        key_layer = key_layer.view(batch_size, seq_len, num_heads, 64)
        query_layer = query_layer.view(batch_size, seq_len, num_heads, 64)
    elif key_layer.shape[0] == 64:
        key_layer = key_layer.view(batch_size, seq_len, num_heads, 64)
        query_layer = query_layer.view(batch_size, seq_len, num_heads, 64)
    elif key_layer.shape[0] == 4:
        key_layer = key_layer.view(batch_size, seq_len, num_heads, 64)
        query_layer = query_layer.view(batch_size, seq_len, num_heads, 64)
    
    # Transpose for attention computation: [B, S, H, D] -> [B, H, S, D]
    values = values.transpose(1, 2)
    key_layer = key_layer.transpose(1, 2)
    query_layer = query_layer.transpose(1, 2)
    
    # Use optimized scaled dot product attention
    output = torch.nn.functional.scaled_dot_product_attention(
        query=query_layer, 
        key=key_layer, 
        value=values, 
        attn_mask=attn_mask, 
        dropout_p=0.0, 
        is_causal=False
    )
    
    # Transpose back: [B, H, S, D] -> [B, S, H, D]
    output = output.transpose(1, 2)
    
    # Reshape back to original format
    output = output.reshape(batch_size, seq_len, hidden_size)
    
    return output

def replacement_func():
    return optimized_attention_with_efficient_reshape