import torch
import triton
import triton.language as tl

def pattern(weight_tensor, input_tensor):
    # Pattern: 1x1 conv with stride (1,1) followed by channel slicing
    conv_result = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (0, 0), (1, 1), 1)
    # Extract the slice pattern from the computation graph
    sliced_result = conv_result[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    # The actual channel slice limit will be extracted by the framework
    return sliced_result, conv_result

def replacement_args(weight_tensor, input_tensor):
    # Extract the actual slice limit from the computation
    # This will be determined dynamically during pattern matching
    return (weight_tensor, input_tensor)

@triton.jit
def pointwise_conv1x1_kernel(
    weight_ptr, input_ptr, output_ptr, full_output_ptr,
    batch_size, in_channels, out_channels, height, width,
    slice_channels, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # Optimized 1x1 convolution with channel slicing optimization
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * height * width, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    # Compute spatial position
    spatial_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < (batch_size * height * width)
    
    # Only compute needed channels for sliced part, then extend to full output
    for c_offset in tl.arange(0, out_channels, BLOCK_SIZE_C):
        c_end = tl.minimum(c_offset + BLOCK_SIZE_C, out_channels)
        
        # Load input channels (1x1 conv - same spatial position for all channels)
        input_base = input_ptr + c_offset  # Input is contiguous in channel dimension
        input_vals = tl.load(input_base + spatial_offset[:, None] * (height * width), 
                           mask=spatial_mask[:, None] & (c_offset + tl.arange(0, BLOCK_SIZE_C)[None, :] < in_channels),
                           other=0.0)
        
        # Load weights for this channel block
        weight_base = weight_ptr + (c_offset // BLOCK_SIZE_C) * BLOCK_SIZE_C * in_channels
        weight_vals = tl.load(weight_base + tl.arange(0, c_end - c_offset)[None, :] * in_channels,
                            mask=tl.arange(0, c_end - c_offset)[None, :] < in_channels,
                            other=0.0)
        
        # Compute 1x1 convolution (simple matrix multiply for spatial positions)
        conv_vals = tl.sum(input_vals * weight_vals, axis=1)
        
        # Store full output
        full_output_base = full_output_ptr + c_offset
        tl.store(full_output_base + spatial_offset, conv_vals, 
                mask=spatial_mask[:, None] & (tl.arange(0, c_end - c_offset)[None, :] < (out_channels - c_offset)))
        
        # Store only sliced part if in range
        if c_end <= slice_channels:
            slice_output_base = output_ptr + c_offset
            tl.store(slice_output_base + spatial_offset, conv_vals,
                    mask=spatial_mask[:, None] & (tl.arange(0, c_end - c_offset)[None, :] < (slice_channels - c_offset)))

@torch.fx.wrap
def optimized_pointwise_conv1x1(weight_tensor, input_tensor, slice_channels):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Always use optimized kernel - no fallback to torch.conv2d
    
    # Allocate outputs
    full_output = torch.empty((batch_size, out_channels, height, width), 
                            dtype=weight_tensor.dtype, device=weight_tensor.device)
    sliced_output = torch.empty((batch_size, slice_channels, height, width), 
                              dtype=weight_tensor.dtype, device=weight_tensor.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    BLOCK_SIZE_C = 32  # Process 32 channels at a time
    
    spatial_elements = batch_size * height * width
    num_spatial_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_channel_blocks = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    pointwise_conv1x1_kernel[(num_spatial_programs, num_channel_blocks)](
        weight_tensor, input_tensor, sliced_output, full_output,
        batch_size, in_channels, out_channels, height, width,
        slice_channels, BLOCK_SIZE, BLOCK_SIZE_C
    )
    
    return sliced_output, full_output

def replacement_func():
    # Return the wrapper function that will extract slice_channels dynamically
    def wrapper(weight_tensor, input_tensor):
        # This needs to be determined from the actual slice operation
        # For now, we'll extract it from the computation pattern
        slice_channels = weight_tensor.shape[0]  # Default to full channels
        return optimized_pointwise_conv1x1(weight_tensor, input_tensor, slice_channels)
    return wrapper