import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_avgpool_kernel(
    input_ptr,  # [batch, input_channels, height, width]
    weight_ptr,  # [output_channels, input_channels, 1, 1]
    output_ptr,  # [batch, output_channels, out_height, out_width]
    batch, input_channels, output_channels, 
    height, width, out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which output location we're computing
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    batch_idx = pid // (output_channels * out_height * out_width)
    channel_idx = (pid % (output_channels * out_height * out_width)) // (out_height * out_width)
    h_idx = (pid % (out_height * out_width)) // out_width
    w_idx = pid % out_width
    
    # For this output position, we need to compute the average of 4 conv results
    # at positions: [h_idx*2, w_idx*2], [h_idx*2, w_idx*2+1], [h_idx*2+1, w_idx*2], [h_idx*2+1, w_idx*2+1]
    
    conv_sum = 0.0
    conv_count = 0
    
    # Compute conv result for each of the 4 input positions and average them
    for dh in range(2):
        for dw in range(2):
            conv_h = h_idx * 2 + dh  # Corresponding conv output position
            conv_w = w_idx * 2 + dw  # Corresponding conv output position
            
            # Only process if within bounds
            if conv_h < height and conv_w < width:
                # Compute 1x1 convolution at this position
                channel_sum = 0.0
                
                # Use fixed-size vector loads to avoid dynamic mask issues
                for c in range(0, input_channels, BLOCK_SIZE):
                    # Calculate actual block size
                    remaining = input_channels - c
                    block_size = BLOCK_SIZE if remaining >= BLOCK_SIZE else remaining
                    
                    # Create offset for this channel block
                    weight_offset = channel_idx * input_channels + c
                    input_offset = batch_idx * input_channels * height * width + \
                                  c * height * width + conv_h * width + conv_w
                    
                    # Load weights and inputs - always use BLOCK_SIZE vector
                    weights = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < block_size)
                    inputs = tl.load(input_ptr + input_offset, mask=tl.arange(0, BLOCK_SIZE) < block_size)
                    
                    # Multiply and accumulate
                    channel_sum += tl.sum(weights * inputs)
                
                # Average over channels for this spatial position
                avg_conv = channel_sum / input_channels
                
                # Accumulate for final average pooling
                if dh == 0 and dw == 0:
                    conv_sum = avg_conv
                    conv_count = 1
                else:
                    conv_sum += avg_conv
                    conv_count += 1
    
    # Final result is average of the conv results
    if conv_count > 0:
        final_output = conv_sum / conv_count
    else:
        final_output = 0.0  # Default for out-of-bound regions
    
    # Store the final result
    output_offset = batch_idx * output_channels * out_height * out_width + \
                    channel_idx * out_height * out_width + \
                    h_idx * out_width + w_idx
    tl.store(output_ptr + output_offset, final_output)

@torch.fx.wrap
def fused_conv2d_avgpool(input, weight):
    # Get input dimensions
    batch, input_channels, height, width = input.shape
    output_channels, _, _, _ = weight.shape
    
    # Calculate output dimensions after avg pooling (stride 2, kernel 2)
    out_height = (height + 1) // 2
    out_width = (width + 1) // 2
    
    # Create output tensor
    output = torch.empty((batch, output_channels, out_height, out_width), 
                        dtype=input.dtype, device=input.device)
    
    # Calculate total number of output elements
    total_elements = batch * output_channels * out_height * out_width
    
    # Launch kernel
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_avgpool_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        batch=batch, input_channels=input_channels, output_channels=output_channels,
        height=height, width=width, out_height=out_height, out_width=out_width,
        BLOCK_SIZE=BLOCK_SIZE,
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
    return fused_conv2d_avgpool