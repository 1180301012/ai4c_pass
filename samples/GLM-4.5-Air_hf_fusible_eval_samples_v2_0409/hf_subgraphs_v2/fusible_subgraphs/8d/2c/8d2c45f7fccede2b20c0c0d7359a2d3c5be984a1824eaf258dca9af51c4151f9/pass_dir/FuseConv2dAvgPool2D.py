import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_avgpool_kernel(
    input_ptr,  # [batch, input_channels, height, width]
    weight_ptr,  # [output_channels, input_channels, kernel_h, kernel_w]
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
    
    # Calculate input coordinates (for 1x1 conv with stride 1)
    input_h = h_idx * 2  # For avg pooling stride 2
    input_w = w_idx * 2  # For avg pooling stride 2
    
    # Load weights for this output channel (1x1 kernel)
    weight_offset = channel_idx * input_channels
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process input channels (1x1 convolution)
    for c in range(0, input_channels, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, input_channels - c)
        weights = tl.load(weight_ptr + weight_offset + c, mask=(c + tl.arange(0, block_size)) < input_channels)
        inputs = tl.load(input_ptr + 
                        batch_idx * input_channels * height * width + 
                        (c + tl.arange(0, block_size)) * height * width + 
                        input_h * width + input_w,
                        mask=(c + tl.arange(0, block_size)) < input_channels)
        
        # Multiply and accumulate (1x1 conv)
        sum_val += weights * inputs
    
    # Average over 2x2 pooling region
    # For locations (h_idx, w_idx), (h_idx, w_idx+1), (h_idx+1, w_idx), (h_idx+1, w_idx+1)
    pool_sum = 0.0
    pool_count = 0
    
    # 2x2 pooling region
    for dh in range(2):
        for dw in range(2):
            pool_h = input_h + dh
            pool_w = input_w + dw
            
            if pool_h < height and pool_w < width:
                # Load all input channels for this spatial location
                for c in range(0, input_channels, BLOCK_SIZE):
                    block_size = min(BLOCK_SIZE, input_channels - c)
                    inputs = tl.load(input_ptr + 
                                    batch_idx * input_channels * height * width + 
                                    (c + tl.arange(0, block_size)) * height * width + 
                                    pool_h * width + pool_w,
                                    mask=(c + tl.arange(0, block_size)) < input_channels)
                    
                    if c == 0:
                        pool_val = inputs[0]  # Initialize with first valid input
                    elif block_size == 1 and inputs[0] != 0:
                        pool_val = inputs[0]  # Use valid input value
                
                    # Only accumulate if the input location is valid
                    if pool_h < height and pool_w < width:
                        if dh == 0 and dw == 0:
                            pool_sum = pool_val
                            pool_count = 1
                        else:
                            pool_sum += pool_val
                            pool_count += 1
    
    # Apply the convolution result (1x1) and then average pooling
    if pool_count > 0:
        output_val = sum_val[0] / pool_count  # conv result + avg pooling
    else:
        output_val = sum_val[0]  # Just conv result if no valid pooling region
    
    # Store the final result
    output_offset = batch_idx * output_channels * out_height * out_width + \
                    channel_idx * out_height * out_width + \
                    h_idx * out_width + w_idx
    tl.store(output_ptr + output_offset, output_val)

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
    BLOCK_SIZE = 128
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