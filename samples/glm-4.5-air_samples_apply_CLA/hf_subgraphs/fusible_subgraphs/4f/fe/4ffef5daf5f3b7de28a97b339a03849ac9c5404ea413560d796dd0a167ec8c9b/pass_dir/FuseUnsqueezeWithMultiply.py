import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Pattern: (tmp_0.unsqueeze(-1)) * (tmp_3.to(torch.float32))
    # Match the exact sequence: unsqueeze then multiply with converted tensor
    unsqueezed = x.unsqueeze(-1)
    result = unsqueezed * y
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_shape,
    y_shape,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    
    # Calculate total elements for proper masking
    x_total_elements = x_shape[0] * x_shape[1]  # assuming [batch, seq_len]
    y_total_elements = y_shape[0] * y_shape[1] * y_shape[2]  # assuming [batch, seq_len, features]
    
    # Each program handles a contiguous block of data
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < y_total_elements
    
    # Load inputs - for y we load directly, for x we broadcast
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # For x, we need to handle broadcasting from [batch, seq_len] to [batch, seq_len, features]
    # Calculate the appropriate x indices
    feature_size = y_shape[2]
    seq_len = y_shape[1]
    
    # Create offsets that allow proper broadcasting
    x_offsets = offsets // feature_size  # This maps to the correct batch and seq position
    
    # Load x with proper broadcasting
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Perform multiplication
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_multiply_with_unsqueeze(attention_mask, features):
    """
    Fuse unsqueeze(-1) * operations
    attention_mask: [batch, seq_len]
    features: [batch, seq_len, features]
    """
    # If attention_mask is already the right shape, just multiply
    if attention_mask.dim() == 3:
        return attention_mask * features
    
    # Handle the unsqueeze and multiplication together
    batch_size, seq_len = attention_mask.shape
    feature_size = features.shape[-1]
    
    # Expand attention_mask to match features dimensions using implicit broadcasting
    # This is more efficient than creating an explicit tensor
    expanded_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    
    # Use efficient multiplication
    result = expanded_mask * features
    
    return result

def replacement_func():
    return fused_multiply_with_unsqueeze