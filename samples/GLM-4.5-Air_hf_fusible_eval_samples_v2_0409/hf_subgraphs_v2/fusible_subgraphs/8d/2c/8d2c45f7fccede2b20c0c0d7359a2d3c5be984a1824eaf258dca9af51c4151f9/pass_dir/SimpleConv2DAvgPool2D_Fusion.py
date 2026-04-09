import torch
import triton
import triton.language as tl

@triton.jit
def simple_fused_kernel(
    input_ptr,  # [batch, input_channels, height, width]
    weight_ptr,  # [output_channels, input_channels, 1, 1]
    output_ptr,  # [batch, output_channels, out_height, out_width]
    batch, input_channels, output_channels, 
    height, width, out_height, out_width,
):
    # Program ID determines which output location we're computing
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    batch_idx = pid // (output_channels * out_height * out_width)
    channel_idx = (pid % (output_channels * out_height * out_width)) // (out_height * out_width)
    h_idx = (pid % (out_height * out_width)) // out_width
    w_idx = pid % out_width
    
    # Compute the average of 4 conv results
    # For each of the 4 input positions that map to this output
    conv_sum = 0.0
    conv_count = 0
    
    for dh in range(2):
        for dw in range(2):
            conv_h = h_idx * 2 + dh
            conv_w = w_idx * 2 + dw
            
            # Only process if within bounds
            if conv_h < height and conv_w < width:
                # Compute 1x1 convolution at this position
                channel_sum = 0.0
                
                # Simple loop over channels (avoid vectorization for now)
                for c in range(input_channels):
                    # Load weight for this channel
                    weight_val = tl.load(weight_ptr + channel_idx * input_channels + c)
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                      batch_idx * input_channels * height * width + 
                                      c * height * width + 
                                      conv_h * width + conv_w)
                    # Add to channel sum
                    channel_sum += weight_val * input_val
                
                # Store the conv result for averaging
                if dh == 0 and dw == 0:
                    conv_sum = channel_sum / input_channels  # Average over channels
                    conv_count = 1
                else:
                    conv_sum += channel_sum / input_channels
                    conv_count += 1
    
    # Final result is average of the conv results
    if conv_count > 0:
        final_output = conv_sum / conv_count
    else:
        final_output = 0.0
    
    # Store the final result
    output_offset = batch_idx * output_channels * out_height * out_width + \
                    channel_idx * out_height * out_width + \
                    h_idx * out_width + w_idx
    tl.store(output_ptr + output_offset, final_output)

@torch.fx.wrap
def simple_fused_conv2d_avgpool(input, weight):
    # Get input dimensions
    batch, input_channels, height, width = input.shape
    output_channels, _, _, _ = weight.shape
    
    # Calculate output dimensions after avg pooling
    out_height = (height + 1) // 2
    out_width = (width + 1) // 2
    
    # Create output tensor
    output = torch.empty((batch, output_channels, out_height, out_width), 
                        dtype=input.dtype, device=input.device)
    
    # Calculate total number of output elements
    total_elements = batch * output_channels * out_height * out_width
    
    # Launch kernel - one program per output element
    num_programs = total_elements if total_elements > 0 else 1
    
    simple_fused_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        batch=batch, input_channels=input_channels, output_channels=output_channels,
        height=height, width=width, out_height=out_height, out_width=out_width,
    )
    
    return output

def pattern(in_0, in_1):
    """Match Conv2D + AvgPool2D pattern"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2

def replacement_args(in_0, in_1):
    """Extract arguments for the fused operation"""
    return (in_1, in_0)  # input first, then weight

def replacement_func():
    """Return the fused conv2d + avgpool function"""
    return simple_fused_conv2d_avgpool