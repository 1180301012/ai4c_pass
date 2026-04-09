import torch
import triton
import triton.language as tl

# Simple pattern to match addition operation
def pattern(x, y):
    """
    Simple pattern matching addition operation - building block for optimization
    """
    result = x + y
    return result

# Argument extraction for replacement
def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr, 
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple addition kernel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_add_wrapper(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Replacement function (returns kernel wrapper)  
def replacement_func():
    return simple_add_wrapper
    input_ptr,  # in_2 - [B, C, H, W] 
    scale_ptr,  # in_0 - [C]
    output_ptr, # tmp_8 - [B, C, H, W]
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID and load tensor coordinates
    pid = tl.program_id(0)
    bid = pid // (channels * height * width // BLOCK_SIZE)
    cid = (pid // (height * width // BLOCK_SIZE)) % channels
    hid = (pid // (width // BLOCK_SIZE)) % height
    wid = (pid % (width // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Skip invalid program IDs
    if bid >= batch_size or cid >= channels or hid >= height or wid.max() >= width:
        return
    
    # Load input features - compute memory offset for current position
    input_offset = (bid * channels + cid) * height * width + hid * width + wid
    features = tl.load(input_ptr + input_offset, mask=wid < width, other=0.0)
    
    # Apply ReLU
    relu_features = tl.maximum(features, 0.0)
    
    # Compute 3x3 average pooling with padding 1
    padding = 1
    kernel_size = 3
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    pool_count = 0
    
    for kh in range(-padding, padding + 1):
        for kw in range(-padding, padding + 1):
            ph = hid + kh
            pw = wid + kw
            
            if 0 <= ph < height and 0 <= pw < width:
                pool_offset = (bid * channels + cid) * height * width + ph * width + pw
                pool_weight = tl.load(input_ptr + pool_offset, mask=pw < width, other=0.0)
                pool_sum += pool_weight
                pool_count += 1
    
    pool_features = pool_sum / pool_count if pool_count > 0 else pool_sum
    
    # Compute residual: (avg_pool - features)
    residual = pool_features - relu_features
    
    # Load scale factor
    scale_offset = cid
    scale_factor = tl.load(scale_ptr + scale_offset, other=1.0)
    
    # Expand scale for broadcasting: [C] -> [H, W] 
    expanded_scale = tl.broadcast_to(scale_factor, [BLOCK_SIZE])
    
    # Apply scaled residual: expanded_scale * residual
    scaled_residual = expanded_scale * residual
    
    # Final feature: relu_features + scaled_residual
    final_features = relu_features + scaled_residual
    
    # Store result
    output_offset = input_offset  # Same spatial layout as input
    tl.store(output_ptr + output_offset, final_features, mask=wid < width)

@torch.fx.wrap
def fused_feature_processing_wrapper(in_0, in_1, in_2):
    """Wrapper function to launch the fused kernel for both outputs"""
    batch_size, channels, height, width = in_2.shape
    
    # Create output tensors
    output_features = torch.empty_like(in_2)  # tmp_8 - processed features
    output_expanded = torch.empty((channels, 1, 1), dtype=in_1.dtype, device=in_1.device)  # tmp_10 - expanded in_1
    
    # Calculate grid dimensions for feature processing
    elements_per_program = 1024  # BLOCK_SIZE
    total_elements = batch_size * channels * height * width
    num_programs = (total_elements + elements_per_program - 1) // elements_per_program
    
    # Launch kernel for feature processing pipeline
    fused_feature_processing_kernel[(num_programs,)](
        input_ptr=in_2,
        scale_ptr=in_0,
        output_ptr=output_features,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=1024,
    )
    
    # Handle simple tensor expansion for second output
    output_expanded_flat = output_expanded.view(-1)
    BLOCK_SIZE_EXP = 256
    num_programs_exp = (channels + BLOCK_SIZE_EXP - 1) // BLOCK_SIZE_EXP
    
    expanded_kernel[(num_programs_exp,)](
        scale_ptr=in_1,
        output_ptr=output_expanded_flat,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE_EXP,
    )
    
    return output_features, output_expanded

# Replacement function (returns kernel wrapper)
def replacement_func():
    return fused_feature_processing_wrapper