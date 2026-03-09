import torch
import triton
import triton.language as tl


def pattern(x, shape, *args):
    # Match the view operation with the actual pattern from the model
    # x is the tensor being viewed, shape is the view shape
    result = x.view(shape)
    return result


def replacement_args(x, shape, *args):
    # Extract the input tensor and view shape
    # x is the input tensor to the view operation
    # shape is the view shape
    return (x, shape)


# Optimized Triton kernel that combines permute + contiguous + view
@triton.jit
def fuse_permute_contiguous_view_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    input_height,
    input_channels,
    input_width,
    output_channels_width,
    output_height,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """Kernel that fuses permute + contiguous + view operations.
    
    Transforms [B, C, H, W] → [B, H, C*W] directly
    """
    batch_idx = tl.program_id(0)
    height_idx = tl.program_id(1)
    
    # Calculate input offsets (original [B, C, H, W] layout)
    x_batch_offset = batch_idx * input_channels * input_height * input_width
    x_hw_offset = height_idx * input_width  # Fixed height position
    
    # Calculate output offsets ([B, H, C*W] layout)
    out_batch_offset = batch_idx * output_height * output_channels_width
    out_hw_offset = height_idx * output_channels_width  # Height in second dimension
    
    # Process each (C,W) position in the output
    for output_cw_idx in range(0, output_channels_width, BLOCK_SIZE_HW):
        # Calculate corresponding position in input
        input_c = output_cw_idx // input_width
        input_w = output_cw_idx % input_width
        
        # Set up memory pointers
        x_ptr_pos = x_batch_offset + input_c * input_height * input_width + x_hw_offset + input_w
        out_ptr_pos = out_batch_offset + out_hw_offset + output_cw_idx
        
        # Create mask for bounds checking
        mask = output_cw_idx + tl.arange(0, BLOCK_SIZE_HW) < output_channels_width
        
        # Load and store the scalar value
        x_val = tl.load(x_ptr + x_ptr_pos, mask=mask, other=0.0)
        tl.store(out_ptr + out_ptr_pos, x_val, mask=mask)


@torch.fx.wrap
def optimized_fused_view(x, view_shape):
    """Optimized fused operation that handles the entire transformation sequence"""
    input_shape = x.shape
    # Use the provided view shape for output
    output_shape = view_shape
    
    batch_size, input_channels, input_height, input_width = input_shape
    output_cw = view_shape[2]  # The third dimension is the combined C*W
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Choose optimal block sizes
    BLOCK_SIZE_HW = 256  # Process multiple channel-width pairs per thread
    
    # Parse view shape to get output dimensions
    output_batch, output_height, output_cw = view_shape
    
    # Calculate grid dimensions
    grid_h = (output_height + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch kernel
    fuse_permute_contiguous_view_kernel[(
        batch_size,
        grid_h
    )](
        x_ptr=x,
        out_ptr=output,
        batch_size=batch_size,
        input_height=input_height,
        input_channels=input_channels,
        input_width=input_width,
        output_channels_width=output_cw,
        output_height=output_height,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return output


def replacement_func():
    return optimized_fused_view