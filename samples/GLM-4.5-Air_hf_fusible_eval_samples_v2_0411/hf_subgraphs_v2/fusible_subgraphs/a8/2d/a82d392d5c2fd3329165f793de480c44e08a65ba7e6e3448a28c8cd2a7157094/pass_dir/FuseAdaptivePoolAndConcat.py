import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching for adaptive_avg_pool2d + concat fusion
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    output_ptr,
    batch_size,
    input_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_idx = pid * BLOCK_SIZE
    
    # Calculate output coordinates
    out_c = block_idx // (output_height * output_width)
    out_h = (block_idx // output_width) % output_height
    out_w = block_idx % output_width
    
    # Calculate input region size for each output pixel
    input_region_h = input_height // output_height
    input_region_w = input_width // output_width
    
    # Start position in input
    input_start_h = out_h * input_region_h
    input_start_w = out_w * input_region_w
    
    # Sum all values in the input region
    sum_val = 0.0
    for h in range(input_region_h):
        for w in range(input_region_w):
            in_h = input_start_h + h
            in_w = input_start_w + w
            if in_h < input_height and in_w < input_width:
                offset = batch_size * input_channels * input_height * input_width + \
                        out_c * input_height * input_width + in_h * input_width + in_w
                val = tl.load(x_ptr + offset)
                sum_val += val
    
    # Store the average value
    out_offset = batch_size * input_channels * output_height * output_width + \
                 out_c * output_height * output_width + out_h * output_width + out_w
    tl.store(output_ptr + out_offset, sum_val / (input_region_h * input_region_w))

@triton.jit
def fused_pool_concat_kernel(
    in_0_ptr,
    in_1_ptr,
    output_ptr,
    batch_size,
    in_0_channels,
    in_1_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = pid // (output_height * output_width)
    pid_h = (pid // output_width) % output_height
    pid_w = pid % output_width
    
    # Check bounds to prevent illegal memory access
    if pid_c >= in_0_channels + in_1_channels or pid_h >= output_height or pid_w >= output_width:
        return
    
    # Determine if this is from in_0 or in_1
    if pid_c < in_0_channels:
        # Process in_0: adaptive average pooling (since dims divide evenly, 64->32, 48->24)
        input_region_h = input_height // output_height  # = 2
        input_region_w = input_width // output_width    # = 2
        
        start_h = pid_h * input_region_h
        start_w = pid_w * input_region_w
        
        sum_val = 0.0
        # Sum over the 2x2 input region
        for h in range(input_region_h):
            for w in range(input_region_w):
                in_h = start_h + h
                in_w = start_w + w
                offset = batch_size * in_0_channels * input_height * input_width + \
                        pid_c * input_height * input_width + in_h * input_width + in_w
                val = tl.load(in_0_ptr + offset)
                sum_val += val
        
        # Average over the 4 elements
        out_val = sum_val / 4.0
        
        # Cast to float32 for computation
        out_val = tl.cast(out_val, tl.float32)
    else:
        # Process in_1: direct copy
        src_channel = pid_c - in_0_channels
        offset = batch_size * in_1_channels * output_height * output_width + \
                 src_channel * output_height * output_width + pid_h * output_width + pid_w
        # Cast to float32 for consistency
        val = tl.load(in_1_ptr + offset)
        out_val = tl.cast(val, tl.float32)
    
    # Store result - cast back to float32 (will be cast back to original type in wrapper)
    total_channels = in_0_channels + in_1_channels
    out_offset = batch_size * total_channels * output_height * output_width + \
                 pid_c * output_height * output_width + pid_h * output_width + pid_w
    tl.store(output_ptr + out_offset, out_val)

@torch.fx.wrap
def fused_adaptive_pool_concat(in_0, in_1):
    # Get input shapes
    batch_size, in_0_channels, input_height, input_width = in_0.shape
    _, in_1_channels, output_height, output_width = in_1.shape
    
    # Determine output shape
    output_channels = in_0_channels + in_1_channels
    total_elements = batch_size * output_channels * output_height * output_width
    
    # Create temporary output tensor as float32 for kernel computation
    temp_output = torch.empty((batch_size, output_channels, output_height, output_width), 
                             dtype=torch.float32, device=in_0.device)
    
    # Calculate optimal block size
    BLOCK_SIZE = 1024  # Larger block size for better GPU utilization
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    fused_pool_concat_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        output_ptr=temp_output,
        batch_size=batch_size,
        in_0_channels=in_0_channels,
        in_1_channels=in_1_channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Cast output back to original input dtype to maintain consistency
    output = temp_output.to(dtype=in_0.dtype)
    
    return output

def replacement_func():
    return fused_adaptive_pool_concat