import torch
import triton
import triton.language as tl

def pattern(x, weight):
    conv_out = torch.conv2d(x, weight, None, (1, 1), (0, 0), (1, 1), 1)
    avg_pool_out = torch.nn.functional.avg_pool2d(conv_out, 2, 2, 0, False, True, None)
    return avg_pool_out

def replacement_args(x, weight):
    return (x, weight)

@triton.jit
def fused_conv2d_avgpool_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    out_height,
    out_width,
):
    # Calculate program indices - one output channel per program
    pid_c = tl.program_id(0)  # output channel
    pid_b = tl.program_id(1)  # batch
    pid_s = tl.program_id(2)  # spatial position
    
    # Check bounds
    if pid_c >= out_channels or pid_b >= batch_size or pid_s >= out_height * out_width:
        return
        
    # Convert spatial position to 2D coordinates
    h_out = pid_s // out_width
    w_out = pid_s % out_width
    
    # Compute input position for 2x2 pooling
    h_in_base = h_out * 2
    w_in_base = w_out * 2
    
    # Initialize accumulator for this output channel
    total_value = 0.0
    count_elements = 0
    
    # Process each input channel and 2x2 spatial window
    for k in range(in_channels):
        for dh in range(2):
            for dw in range(2):
                input_h = h_in_base + dh
                input_w = w_in_base + dw
                
                if input_h < in_height and input_w < in_width:
                    # Load input for this batch, channel, and position
                    input_idx = (pid_b * in_channels + k) * in_height * in_width + input_h * in_width + input_w
                    input_val = tl.load(input_ptr + input_idx)
                    
                    # Load weight for this output channel and input channel
                    weight_idx = pid_c * in_channels + k
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate
                    total_value += weight_val * input_val
                    count_elements += 1
    
    # Compute average
    result = total_value / count_elements if count_elements > 0 else total_value
    
    # Store result
    output_idx = pid_b * out_channels * out_height * out_width + pid_c * out_height * out_width + pid_s
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_conv2d_avgpool(x, weight):
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # After avg pooling with 2x2 stride=2
    out_height = in_height // 2
    out_width = in_width // 2
    
    # Reshape output to [batch, out_channels, out_height, out_width]
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                        dtype=x.dtype, device=x.device)
    
    # Flatten output spatial dimensions for kernel processing
    output_flat = output.reshape(batch_size, out_channels, out_height * out_width)
    
    # Calculate grid dimensions - one program per output channel, batch, and spatial position
    grid_c = out_channels
    grid_b = batch_size
    grid_s = out_height * out_width  # one spatial position per program
    
    # Launch kernel
    fused_conv2d_avgpool_kernel[(grid_c, grid_b, grid_s)](
        input_ptr=x,
        weight_ptr=weight,
        output_ptr=output_flat,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
    )
    
    return output

def replacement_func():
    return fused_conv2d_avgpool