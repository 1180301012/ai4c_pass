import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matches adaptive_avg_pool2d followed by channel concatenation
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    """
    Extract arguments for the fused kernel
    """
    return (in_0, in_1)

@triton.jit
def fused_adaptive_pool_concat_kernel(
    input_ptr,
    input2_ptr,
    output_ptr,
    batch_size,
    input_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    input2_channels,
    output_channels,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Efficient fused kernel that performs adaptive average pooling followed by concatenation
    """
    # Get program indices (3D grid maximum)
    b = tl.program_id(0)                    # batch
    c_start = tl.program_id(1) * BLOCK_SIZE_C  # channel block
    hw_idx = tl.program_id(2) * BLOCK_SIZE_HW  # flattened spatial block
    
    # Convert flattened index to 2D spatial coordinates
    h = hw_idx // output_width
    w = hw_idx % output_width
    
    # Process multiple channels per thread for better efficiency
    for c_offset in range(0, BLOCK_SIZE_C):
        c_out = c_start + c_offset
        
        # Only process channels within bounds
        if c_out < output_channels:
            # Process this channel if in spatial bounds
            if h < output_height and w < output_width:
                out_idx = (b * output_channels + c_out) * output_height * output_width + h * output_width + w
                
                if c_out < input_channels:
                    # Optimized adaptive pooling
                    x_start_h = tl.cast(input_height * h / output_height, tl.int32)
                    x_end_h = tl.cast(input_height * (h + 1) / output_height, tl.int32)
                    x_start_w = tl.cast(input_width * w / output_width, tl.int32)
                    x_end_w = tl.cast(input_width * (w + 1) / output_width, tl.int32)
                    
                    # More efficient accumulation
                    sum_val = 0.0
                    count = 0
                    end_h = tl.minimum(x_end_h, input_height)
                    end_w = tl.minimum(x_end_w, input_width)
                    
                    # Direct accumulation with single bounds check
                    for y_h in range(x_start_h, end_h):
                        for y_w in range(x_start_w, end_w):
                            idx = (b * input_channels + c_out) * input_height * input_width + y_h * input_width + y_w
                            sum_val += tl.load(input_ptr + idx)
                            count += 1
                    
                    # Store averaged result
                    if count > 0:
                        tl.store(output_ptr + out_idx, sum_val / count)
                    else:
                        tl.store(output_ptr + out_idx, 0.0)
                else:
                    # Direct copy from second input (highly efficient)
                    c_in2 = c_out - input_channels
                    if c_in2 < input2_channels:
                        in_idx = (b * input2_channels + c_in2) * output_height * output_width + h * output_width + w
                        value = tl.load(input2_ptr + in_idx)
                        tl.store(output_ptr + out_idx, value)
                    else:
                        tl.store(output_ptr + out_idx, 0.0)

@torch.fx.wrap
def fused_adaptive_pool_concat(in_0, in_1):
    """
    Wrapper function for the fused adaptive pooling and concatenation
    """
    batch_size = in_0.size(0)
    input_channels = in_0.size(1)
    input_height = in_0.size(2)
    input_width = in_0.size(3)
    output_height = 32
    output_width = 24
    input2_channels = in_1.size(1)
    output_channels = input_channels + input2_channels
    
    # Create output tensor
    output = torch.empty((batch_size, output_channels, output_height, output_width), 
                        dtype=in_0.dtype, device=in_0.device)
    
    # Use block sizes that provide good GPU utilization
    BLOCK_SIZE_C = 64   # Block size for channels
    BLOCK_SIZE_HW = 32   # Block size for flattened spatial dimensions
    
    # Calculate grid dimensions for 3D launch (batch, channel_blocks, spatial_blocks)
    num_batch = batch_size
    num_channels = (output_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_hw_blocks = (output_height * output_width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch kernel with 3D grid (batch, channel_blocks, spatial_blocks)
    fused_adaptive_pool_concat_kernel[(num_batch, num_channels, num_hw_blocks)](
        in_0,
        in_1,
        output,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        input2_channels,
        output_channels,
        BLOCK_SIZE_C,
        BLOCK_SIZE_HW,
    )
    
    return output

def replacement_func():
    """
    Returns the fused function
    """
    return fused_adaptive_pool_concat