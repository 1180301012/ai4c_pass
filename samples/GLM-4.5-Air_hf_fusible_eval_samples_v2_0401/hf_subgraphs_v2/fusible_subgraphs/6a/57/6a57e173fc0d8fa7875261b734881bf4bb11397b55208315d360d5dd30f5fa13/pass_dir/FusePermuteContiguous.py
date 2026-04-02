import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Match permute(0, 2, 1, 3) + contiguous pattern
    This pattern reorders dimensions from [B, C, H, W] to [B, H, C, W]
    """
    permuted = input_tensor.permute(0, 2, 1, 3)
    result = permuted.contiguous()
    return result

def replacement_args(input_tensor):
    """Extract arguments for the fused permute + contiguous operation"""
    return (input_tensor,)

@triton.jit
def fused_permute_contiguous_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that fuses permute(0, 2, 1, 3) and contiguous operations
    Input: [B, C, H, W] -> Output: [B, H, C, W]
    """
    pid = tl.program_id(0)
    
    # Calculate total number of elements
    total_elements = batch_size * in_height * in_channels * in_width
    
    # Each program handles BLOCK_SIZE elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert linear offset to input coordinates [B, C, H, W]
    # B = offset / (C * H * W)
    # C = (offset % (C * H * W)) / (H * W)
    # H = (offset % (H * W)) / W
    # W = offset % W
    
    input_coord_offset = offsets
    input_coord_total_size = in_channels * in_height * in_width
    
    batch_idx = input_coord_offset // input_coord_total_size
    remaining = input_coord_offset % input_coord_total_size
    
    in_channels_coord_total_size = in_height * in_width
    in_channels_idx = remaining // in_channels_coord_total_size
    remaining = remaining % in_channels_coord_total_size
    
    in_height_idx = remaining // in_width
    in_width_idx = remaining % in_width
    
    # Calculate output coordinates [B, H, C, W]
    # The output has dimensions: [B, H, C, W]
    output_coord_total_size = in_channels * in_width
    output_coord = batch_idx * in_height * output_coord_total_size + \
                   in_height_idx * output_coord_total_size + \
                   in_channels_idx * in_width + \
                   in_width_idx
    
    # Load from input and store to output
    input_value = tl.load(input_ptr + input_coord_offset, mask=mask)
    tl.store(output_ptr + output_coord, input_value, mask=mask)

@torch.fx.wrap
def fused_permute_contiguous(input_tensor):
    """
    Fused permute(0, 2, 1, 3) + contiguous operation using Triton
    
    Args:
        input_tensor: Input tensor with shape [B, C, H, W]
    
    Returns:
        Output tensor with shape [B, H, C, W] in contiguous layout
    """
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    
    # Create output tensor with correct shape [B, H, C, W]
    output_shape = (batch_size, in_height, in_channels, in_width)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate total elements and launch kernel
    total_elements = batch_size * in_channels * in_height * in_width
    BLOCK_SIZE = 1024  # Good trade-off between occupancy and granularity
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_permute_contiguous_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    """Return the fused permute + contiguous function"""
    return fused_permute_contiguous