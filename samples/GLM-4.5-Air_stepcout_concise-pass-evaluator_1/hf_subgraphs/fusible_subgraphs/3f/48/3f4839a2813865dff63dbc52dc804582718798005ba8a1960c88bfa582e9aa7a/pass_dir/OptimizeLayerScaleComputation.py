import torch
import triton
import triton.language as tl

def pattern(layer_scale, residual, feature_maps):
    """
    Pattern: Layer scale computation with sequential unsqueeze operations
    Original: 
        tmp_4 = layer_scale.unsqueeze(-1)  # [48] → [48, 1]
        tmp_5 = tmp_4.unsqueeze(-1)         # [48, 1] → [48, 1, 1]  
        tmp_6 = tmp_5 * residual            # [48, 1, 1] * [B, C, H, W] → [B, C, H, W]
        result = feature_maps + tmp_6       # residual addition
    """
    tmp_4 = layer_scale.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 * residual
    result = feature_maps + tmp_6
    return result

def replacement_args(layer_scale, residual, feature_maps):
    return (layer_scale, residual, feature_maps)

@triton.jit
def layer_scale_kernel(
    layer_scale_ptr,
    residual_ptr,
    feature_maps_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load layer scale (expanded to full size)
    layer_scale_idx = offsets // (height * width)  # Group by channel
    layer_scale_val = tl.load(layer_scale_ptr + layer_scale_idx % channels, mask=layer_scale_idx < channels)
    
    # Load residual and feature maps
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    feature_maps = tl.load(feature_maps_ptr + offsets, mask=mask, other=0.0)
    
    # Layer scale computation: multiply scaling factor with residual
    scaled_residual = layer_scale_val * residual
    
    # Add to feature maps
    output = feature_maps + scaled_residual
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_layer_scale_forward(layer_scale, residual, feature_maps):
    # Get tensor shapes
    batch_size, channels, height, width = feature_maps.shape
    
    # Calculate total number of elements
    total_elements = batch_size * channels * height * width
    
    # Set Triton kernel parameters
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(feature_maps)
    
    # Launch kernel
    layer_scale_kernel[(num_programs,)](
        layer_scale_ptr=layer_scale,
        residual_ptr=residual,
        feature_maps_ptr=feature_maps,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_layer_scale_forward