import torch
import triton
import triton.language as tl

def pattern(weight_tensor, input_tensor):
    # Pattern: strided conv followed by channel slicing
    conv_result = torch.conv2d(input_tensor, weight_tensor, None, (2, 2), (0, 0), (1, 1), 1)
    # The actual channel slice limit will be extracted by the framework
    sliced_result = conv_result[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    return sliced_result, conv_result

def replacement_args(weight_tensor, input_tensor):
    return (weight_tensor, input_tensor)

@triton.jit
def strided_conv2d_kernel(
    weight_ptr, input_ptr, output_ptr, full_output_ptr,
    batch_size, in_channels, out_channels, in_height, in_width,
    out_height, out_width, slice_channels,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # Optimized strided 2D convolution with stride (2,2)
    pid = tl.program_id(0)
    num_programs = tl.cdiv(out_height * out_width, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    # Compute output spatial position
    out_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_y = out_offset // out_width
    out_x = out_offset % out_width
    out_mask = out_offset < (out_height * out_width)
    
    # Convert to input coordinates (with stride 2)
    in_y = out_y * 2  # stride of 2
    in_x = out_x * 2  # stride of 2
    
    # Process output channels in blocks
    for c_offset in tl.arange(0, out_channels, BLOCK_SIZE_C):
        c_end = tl.minimum(c_offset + BLOCK_SIZE_C, out_channels)
        c_mask = tl.arange(0, c_end - c_offset) < (out_channels - c_offset)
        
        # Initialize output accumulation for this channel block
        output_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # For 1x1 convolution, no spatial window needed
        # Load input at the specific spatial position
        spatial_pos = in_y * in_width + in_x
        input_base_ptr = input_ptr + spatial_pos * in_channels
        
        # Load input channels for this spatial position
        input_channels = tl.load(input_base_ptr + tl.arange(0, in_channels),
                               mask=tl.arange(0, in_channels) < in_channels,
                               other=0.0)
        
        # Load corresponding weights for this output channel block
        weight_base_ptr = weight_ptr + c_offset * in_channels
        weight_channels = tl.load(weight_base_ptr + tl.arange(0, (c_end - c_offset) * in_channels),
                                mask=tl.arange(0, (c_end - c_offset) * in_channels) < ((c_end - c_offset) * in_channels),
                                other=0.0)
        weight_channels = weight_channels.reshape((c_end - c_offset), in_channels)
        
        # Compute dot product for each output channel
        for oc in range(c_end - c_offset):
            if oc < (c_end - c_offset):
                output_vals += weight_channels[oc, :] * input_channels
        
        # Store full output
        full_output_base = full_output_ptr + (c_offset + out_offset * out_channels)
        for oc in range(c_end - c_offset):
            if out_mask[oc] and c_mask[oc]:
                tl.store(full_output_base + oc * out_channels, output_vals[oc])
        
        # Store only sliced part if in range
        if c_end <= slice_channels:
            slice_output_base = output_ptr + (c_offset + out_offset * slice_channels)
            for oc in range(c_end - c_offset):
                if out_mask[oc] and c_mask[oc]:
                    tl.store(slice_output_base + oc * slice_channels, output_vals[oc])

@torch.fx.wrap
def optimized_strided_conv2d(weight_tensor, input_tensor, slice_channels):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Calculate output dimensions for stride (2,2)
    out_height = (in_height - 1) // 2 + 1  # No padding, stride 2
    out_width = (in_width - 1) // 2 + 1
    
    # Allocate outputs
    full_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                            dtype=weight_tensor.dtype, device=weight_tensor.device)
    sliced_output = torch.empty((batch_size, slice_channels, out_height, out_width), 
                              dtype=weight_tensor.dtype, device=weight_tensor.device)
    
    # Launch kernel for strided convolution
    BLOCK_SIZE = 512
    BLOCK_SIZE_C = 32
    
    spatial_elements = batch_size * out_height * out_width
    num_spatial_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_channel_blocks = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    strided_conv2d_kernel[(num_spatial_programs, num_channel_blocks)](
        weight_tensor, input_tensor, sliced_output, full_output,
        batch_size, in_channels, out_channels, in_height, in_width,
        out_height, out_width, slice_channels, BLOCK_SIZE, BLOCK_SIZE_C
    )
    
    return sliced_output, full_output

def replacement_func():
    def wrapper(weight_tensor, input_tensor):
        slice_channels = weight_tensor.shape[0]  # Will be optimized by framework
        return optimized_strided_conv2d(weight_tensor, input_tensor, slice_channels)
    return wrapper