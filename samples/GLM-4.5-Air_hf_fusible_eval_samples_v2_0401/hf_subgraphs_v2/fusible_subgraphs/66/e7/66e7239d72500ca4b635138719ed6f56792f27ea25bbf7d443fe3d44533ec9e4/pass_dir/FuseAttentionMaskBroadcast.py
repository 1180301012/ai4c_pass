import torch

@torch.fx.wrap  
def fused_attention_mask_broadcast(attention_mask, layer_norm_output):
    """
    Fusion of: unsqueeze(-1) → expand_as → float() operations
    Using efficient PyTorch operations instead of custom Triton kernel
    """
    # Use efficient PyTorch operations for broadcasting
    # This avoids the overhead of a custom Triton kernel
    
    # Step 1: Reshape attention_mask to add the feature dimension
    # attention_mask: [batch, seq_len] → [batch, seq_len, 1]
    unsqueezed = attention_mask.unsqueeze(-1)
    
    # Step 2: Broadcast to match layer_norm_output shape
    # This uses PyTorch's highly optimized broadcasting
    expanded = unsqueezed.expand_as(layer_norm_output)
    
    # Step 3: Convert to float32
    # Use PyTorch's efficient type conversion
    result = expanded.float()
    
    return result

def pattern(attention_mask, layer_norm_output):
    """Match the sequence: unsqueeze(-1) → expand_as → float()"""
    # This matches the pattern: attention_mask.unsqueeze(-1).expand_as(layer_norm_output).float()
    unsqueezed = attention_mask.unsqueeze(-1)
    expanded = unsqueezed.expand_as(layer_norm_output)
    float_converted = expanded.float()
    return float_converted

def replacement_args(attention_mask, layer_norm_output):
    """Extract arguments for replacement function"""
    return (attention_mask, layer_norm_output)

def replacement_func():
    """Return the fused function implementation"""
    return fused_attention_mask_broadcast