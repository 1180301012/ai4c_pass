import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def adaptive_pool_concat_kernel(
    input_ptr,
    concat_ptr,
    output_ptr,
    batch_size,
    input_channels,
    input_height,
    input_width,
    concat_channels,
    output_height,
    output_width,
):
    # Each program handles one output element using 3D grid
    # Flatten batch and channel dimensions: pid_bc = pid_b * input_channels + pid_c
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1) 
    pid_w = tl.program_id(2)
    
    # Unflatten batch and channel from linear index
    pid_b = pid_bc // input_channels
    pid_c = pid_bc % input_channels
    
    # Check bounds (split chained boolean operators)
    if ((pid_b >= batch_size) or (pid_c >= input_channels) or (pid_h >= output_height) or (pid_w >= output_width)):
        return
    
    # Calculate adaptive pooling averaging for this position
    val_sum = 0.0
    val_count = 0
    
    # Check all 4 positions in the 2x2 input region
    for dh in range(2):
        for dw in range(2):
            input_h = pid_h * 2 + dh
            input_w = pid_w * 2 + dw
            
            if input_h < input_height and input_w < input_width:
                # Calculate linear offset in input tensor
                input_offset = (pid_b * input_channels * input_height * input_width + 
                              input_h * input_channels * input_width + 
                              input_w * input_channels + pid_c)
                
                val_ptr = input_ptr + input_offset
                val_sum += tl.load(val_ptr, other=0.0)
                val_count += 1
    
    # Store pooled result in output (only for input channels)
    if val_count > 0:
        pooled_val = val_sum / val_count
        output_pooled_offset = (pid_b * input_channels * output_height * output_width + 
                               pid_c * output_height * output_width + 
                               pid_h * output_width + pid_w)
        output_ptr_pooled = output_ptr + output_pooled_offset
        tl.store(output_ptr_pooled, pooled_val)

@torch.fx.wrap
def adaptive_pool_concat_fused(in_0, in_1):
    batch_size = in_0.shape[0]
    pooled_channels = in_0.shape[1]
    original_height = in_0.shape[2]
    original_width = in_0.shape[3]
    concat_channels = in_1.shape[1]
    output_height = 32
    output_width = 24
    
    # Create output tensor
    output_shape = (batch_size, pooled_channels + concat_channels, output_height, output_width)
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Phase 1: Compute adaptive pooling for input channels
    # Use 3D grid: (batch * channels, height, width)
    grid = (
        batch_size * pooled_channels,  # flattened batch * channels
        output_height,                  # output height
        output_width                    # output width
    )
    
    # Launch kernel for pooling computation
    adaptive_pool_concat_kernel[grid](
        in_0,
        in_1,
        output,
        batch_size,
        pooled_channels,
        original_height,
        original_width,
        concat_channels,
        output_height,
        output_width,
    )
    
    # Phase 2: Copy concatenation channels from in_1 to output
    # Simple slice assignment - in_1 already has the correct shape (32, 24)
    output[:, pooled_channels:, :, :] = in_1
    
    return output

def replacement_func():
    return adaptive_pool_concat_fused