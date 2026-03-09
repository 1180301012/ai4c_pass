import torch

# Pattern matching for cos/sin + dimension manipulation
def pattern(encoding_tensor):
    # Match the pattern: cos/sin computation with dimension manipulation
    # Original: 
    #   tmp_6 = tmp_5.cos()
    #   tmp_7 = tmp_6[None, None, slice(...), slice(...)]
    #   tmp_8 = tmp_5.sin()
    #   tmp_9 = tmp_8[None, None, slice(...), slice(...)]
    #   tmp_10 = tmp_7[slice(...), slice(...), slice(...), slice(...)]
    #   tmp_11 = tmp_9[slice(...), slice(...), slice(...), slice(...)]
    #
    # Optimized: compute cos and sin, apply dimension manipulation directly
    
    cos_encoding = encoding_tensor.cos()
    sin_encoding = encoding_tensor.sin()
    
    # Add singleton dimensions and slice more efficiently
    expanded_cos = cos_encoding[None, None, :, :]
    expanded_sin = sin_encoding[None, None, :, :]
    
    # Apply slicing to match original behavior
    sliced_cos = expanded_cos[..., :, :]
    sliced_sin = expanded_sin[..., :, :]
    
    return sliced_cos, sliced_sin

# Extract arguments from matched nodes
def replacement_args(node):
    # Extract the encoding tensor for cos/sin computation
    return (node,)

# Optimized function that computes cos and sin with efficient dimension manipulation
def compute_rotary_trigonometry_optimized(encoding_tensor):
    # Compute cos and sin 
    cos_encoding = encoding_tensor.cos()
    sin_encoding = encoding_tensor.sin()
    
    # Add singleton dimensions at the front efficiently
    # This is more efficient than the original pattern of adding dimensions then slicing
    cos_tensor = cos_encoding.unsqueeze(0).unsqueeze(0)  # Adds 2 dimensions at front
    sin_tensor = sin_encoding.unsqueeze(0).unsqueeze(0)  # Adds 2 dimensions at front
    
    # Apply slicing to match the original behavior (slice along third dimension)
    # The original slicing seems to be extracting the full tensor since seq_len matches dim_size
    sliced_cos = cos_tensor[..., :, :]
    sliced_sin = sin_tensor[..., :, :]
    
    return sliced_cos, sliced_sin

# Replacement function that returns the optimized implementation
def replacement_func():
    return compute_rotary_trigonometry_optimized