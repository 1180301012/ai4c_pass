import torch
import triton
import triton.language as tl

def pattern(conv_input, weight_bias):
    weight, bias = weight_bias
    tmp_2 = torch.conv2d(conv_input, weight, bias, (1, 1), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(640, 640), mode='bilinear')
    return tmp_3,

def replacement_args(conv_input, weight_bias):
    return (conv_input, weight_bias)

@triton.jit
def fused_conv_interpolate_kernel(
    input_ptr, weight_ptr, bias_ptr,  
    output_ptr,
    batch_size, in_channels, in_h, in_w,
    out_channels, kernel_h, kernel_w,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Conv2D parameters
    pad_h, pad_w = 1, 1
    
    # Grid for output pixels
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output coordinates
    out_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_x = out_x % out_w
    out_y = out_y % out_h
    
    # Load bias
    bias = tl.load(bias_ptr)
    
    # Process each output pixel
    for i, x in enumerate(out_x):
        for j, y in enumerate(out_y):
            # Calculate input coordinate considering padding
            in_x = x - pad_w
            in_y = y - pad_h
            
            # Skip if out of bounds after padding
            if in_x < 0 or in_x >= in_w or in_y < 0 or in_y >= in_h:
                # Set to zero if out of bounds
                for oc in range(out_channels):
                    output_idx = ((pid_b * out_channels + oc) * out_h + y) * out_w + x
                    tl.store(output_ptr + output_idx, bias[oc])
                continue
            
            # Conv2D computation
            acc = bias
            
            # Load input neighborhood
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    in_y_idx = in_y + ki
                    in_x_idx = in_x + kj
                    
                    if 0 <= in_y_idx < in_h and 0 <= in_x_idx < in_w:
                        input_idx = ((pid_b * in_channels + in_channels//out_channels) * in_h + in_y_idx) * in_w + in_x_idx
                        input_val = tl.load(input_ptr + input_idx)
                        
                        weight_idx = (in_channels//out_channels * kernel_h + ki) * kernel_w + kj
                        weight_val = tl.load(weight_ptr + weight_idx)
                        
                        acc += input_val * weight_val
                    
            # Store output value
            for oc in range(out_channels):
                output_idx = ((pid_b * out_channels + oc) * out_h + y) * out_w + x
                tl.store(output_ptr + output_idx, acc[oc])

@torch.fx.wrap  
def fused_conv_interpolate(input, weight_bias):
    weight, bias = weight_bias
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h, out_w = 640, 640
    
    output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=input.dtype, device=input.device)
    
    # Adjust grid and block sizes based on output dimensions
    BLOCK_SIZE = 8
    grid_x = (out_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (out_h + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_b = batch_size
    grid = (grid_x, grid_y, grid_b)
    
    fused_conv_interpolate_kernel[grid](
        input, weight, bias,
        output,
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv_interpolate