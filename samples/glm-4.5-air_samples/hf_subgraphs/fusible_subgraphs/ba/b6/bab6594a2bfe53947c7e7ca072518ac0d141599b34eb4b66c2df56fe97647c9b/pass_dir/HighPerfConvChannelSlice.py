import torch
import triton
import triton.language as tl

def pattern(weight_tensor, input_tensor):
    # Pattern: general conv followed by channel slicing
    conv_result = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (0, 0), (1, 1), 1)
    sliced_result = conv_result[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    return sliced_result, conv_result

def replacement_args(weight_tensor, input_tensor):
    return (weight_tensor, input_tensor)

@triton.jit
def high_perf_conv_kernel(
    weight_ptr, input_ptr, output_ptr, full_output_ptr,
    batch_size, in_channels, out_channels, height, width,
    stride_h, stride_w, slice_channels,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # High-performance convolution with configurable stride
    pid = tl.program_id(0)
    grid_size = tl.cdiv(height * width, BLOCK_SIZE)
    
    if pid >= grid_size:
        return
    
    # Compute spatial position in output
    spatial_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < (height * width)
    
    # Convert to output coordinates
    out_y = spatial_offset // width
    out_x = spatial_offset % width
    
    # For 1x1 conv with stride, compute input coordinates
    in_y = out_y * stride_h
    in_x = out_x * stride_w
    
    # Only proceed if input coordinates are valid
    spatial_valid_mask = (in_y < height) & (in_x < width)
    
    # Process output channels in blocks
    for c_offset in tl.arange(0, out_channels, BLOCK_SIZE_C):
        c_end = tl.minimum(c_offset + BLOCK_SIZE_C, out_channels)
        c_mask = tl.arange(0, c_end - c_offset) < (out_channels - c_offset)
        
        # For valid spatial positions, compute convolution
        if tl.any(spatial_valid_mask):
            # Get input position for 1x1 conv
            valid_spatial = spatial_valid_mask
            input_pos = tl.where(valid_spatial, in_y * width + in_x, 0)
            
            # Load input channels for valid positions
            input_channels = tl.zeros((BLOCK_SIZE, in_channels), dtype=tl.float32)
            if tl.any(valid_spatial):
                input_base = input_ptr + input_pos[valid_spatial] * in_channels
                input_channels = tl.load(input_base + tl.arange(0, in_channels)[None, :],
                                       mask=valid_spatial[:, None] & (tl.arange(0, in_channels)[None, :] < in_channels),
                                       other=0.0)
            
            # Load weight block
            weight_base = weight_ptr + c_offset * in_channels
            weight_channels = tl.load(weight_base + tl.arange(0, (c_end - c_offset) * in_channels),
                                    mask=tl.arange(0, (c_end - c_offset) * in_channels) < ((c_end - c_offset) * in_channels),
                                    other=0.0)
            weight_channels = weight_channels.reshape((c_end - c_offset), in_channels)
            
            # Compute convolution: output = input * weights
            output_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for oc in range(c_end - c_offset):
                if tl.any(valid_spatial):
                    # Vectorized dot product
                    output_vals += tl.sum(input_channels * weight_channels[oc, :], axis=1)
        
        else:
            output_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Store full output with masking
        for oc in range(c_end - c_offset):
            if spatial_mask[oc] and c_mask[oc]:
                full_output_idx = (c_offset + oc) + spatial_offset * out_channels
                tl.store(full_output_ptr + full_output_idx, output_vals[oc])
                
                # Store sliced output if in range
                if (c_offset + oc) < slice_channels:
                    slice_output_idx = (c_offset + oc) + spatial_offset * slice_channels
                    tl.store(output_ptr + slice_output_idx, output_vals[oc])

@torch.fx.wrap
def optimized_high_perf_conv(weight_tensor, input_tensor, slice_channels):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Use high-performance autotuned strategy
    # For 1x1 convolutions with slicing, we want to avoid computing all channels
    # when only a small subset is needed
    slice_ratio = slice_channels / out_channels
    
    if slice_ratio < 0.7:  # Only optimize if we save significant computation
        # Allocate outputs
        full_output = torch.empty((batch_size, out_channels, height, width), 
                                dtype=weight_tensor.dtype, device=weight_tensor.device)
        sliced_output = torch.empty((batch_size, slice_channels, height, width), 
                                  dtype=weight_tensor.dtype, device=weight_tensor.device)
        
        # Launch optimized kernel
        BLOCK_SIZE = 512  # Optimized for spatial locality
        BLOCK_SIZE_C = 64  # Process 64 channels at a time for better occupancy
        
        spatial_elements = batch_size * height * width
        num_spatial_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_channel_blocks = 1  # Single launch handles all channels
        
        high_perf_conv_kernel[(num_spatial_programs, num_channel_blocks)](
            weight_tensor, input_tensor, sliced_output, full_output,
            batch_size, in_channels, out_channels, height, width,
            1, 1, slice_channels, BLOCK_SIZE, BLOCK_SIZE_C
        )
        
        return sliced_output, full_output
    else:
        # Always use optimized kernel even when most channels are needed
        # Allocate outputs  
        full_output = torch.empty((batch_size, out_channels, height, width), 
                                dtype=weight_tensor.dtype, device=weight_tensor.device)
        sliced_output = torch.empty((batch_size, slice_channels, height, width), 
                                  dtype=weight_tensor.dtype, device=weight_tensor.device)
        
        # Launch optimized kernel with different configuration
        BLOCK_SIZE = 256  # Larger block size for when we need more channels
        BLOCK_SIZE_C = 128  # Process more channels together
        
        spatial_elements = batch_size * height * width
        num_spatial_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_channel_blocks = 1
        
        high_perf_conv_kernel[(num_spatial_programs, num_channel_blocks)](
            weight_tensor, input_tensor, sliced_output, full_output,
            batch_size, in_channels, out_channels, height, width,
            1, 1, slice_channels, BLOCK_SIZE, BLOCK_SIZE_C
        )
        
        return sliced_output, full_output

def replacement_func():
    def wrapper(weight_tensor, input_tensor):
        slice_channels = weight_tensor.shape[0]  # Will be optimized by framework
        return optimized_high_perf_conv(weight_tensor, input_tensor, slice_channels)
    return wrapper