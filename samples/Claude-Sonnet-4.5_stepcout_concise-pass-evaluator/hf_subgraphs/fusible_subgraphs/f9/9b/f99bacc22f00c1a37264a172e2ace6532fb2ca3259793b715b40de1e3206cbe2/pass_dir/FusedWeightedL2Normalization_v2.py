import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match: ReLU -> Flatten -> L2-Norm -> Scale -> Clamp -> Divide -> Multiply
    This is a weighted L2 normalization operation with scale 0.07216878364870322.
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * tmp_0
    return (tmp_7,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, 0.07216878364870322)


@triton.jit
def fused_weighted_l2_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    num_channels,
    spatial_size,
    scale,
    min_clamp,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for weighted L2 normalization.
    Each program handles one (batch, channel) combination.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // num_channels
    channel_idx = pid % num_channels
    
    # Base offset for this (batch, channel) pair
    base_offset = batch_idx * num_channels * spatial_size + channel_idx * spatial_size
    
    # Compute L2 norm for this (batch, channel) pair
    norm_squared = 0.0
    for block_start in range(0, spatial_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load input and apply ReLU
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        vals = tl.maximum(vals, 0.0)
        
        # Accumulate squared values
        norm_squared += tl.sum(vals * vals, axis=0)
    
    # Compute normalized scale factor
    norm = tl.sqrt(norm_squared)
    scaled_norm = norm * scale
    clamped_norm = tl.maximum(scaled_norm, min_clamp)
    
    # Load weight (scalar broadcast)
    weight = tl.load(weight_ptr)
    
    # Normalize and scale by weight
    norm_factor = weight / clamped_norm
    
    # Write normalized and weighted output
    for block_start in range(0, spatial_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load input and apply ReLU
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        vals = tl.maximum(vals, 0.0)
        
        # Apply normalization and weighting
        output = vals * norm_factor
        
        # Store result
        tl.store(output_ptr + base_offset + offsets, output, mask=mask)


@torch.fx.wrap
def fused_weighted_l2_norm(weight, input_tensor, scale):
    """
    Wrapper function for the fused weighted L2 normalization kernel.
    
    Args:
        weight: Scalar weight tensor [1]
        input_tensor: Input tensor [B, C, H, W]
        scale: Scale factor to apply to norm
    """
    # Get input shape
    B, C, H, W = input_tensor.shape
    spatial_size = H * W
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    grid = (B * C,)
    BLOCK_SIZE = 256
    
    fused_weighted_l2_norm_kernel[grid](
        input_tensor,
        weight,
        output,
        B,
        C,
        spatial_size,
        scale,
        1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_weighted_l2_norm