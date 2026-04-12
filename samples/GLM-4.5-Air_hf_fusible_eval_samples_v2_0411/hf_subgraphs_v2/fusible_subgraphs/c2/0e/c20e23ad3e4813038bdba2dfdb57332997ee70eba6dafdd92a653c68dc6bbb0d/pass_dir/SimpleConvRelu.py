import torch
import triton
import triton.language as tl

# Pattern matching function - matches conv2d + ReLU
def pattern(in_0, in_1, in_3):
    # Simplified pattern: just conv2d + ReLU
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    relu_out = torch.nn.functional.relu(conv2d, inplace=True)
    return (relu_out,)  # Return as tuple

# Argument extraction function
def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

# Optimized kernel for conv2d + ReLU fusion
@triton.jit
def conv_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE output positions
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate dimensions for each output position
    output_h = output_idx // (width_out * out_channels * batch_size)
    pos_in_row = output_idx % (width_out * out_channels * batch_size)
    output_w = pos_in_row // (out_channels * batch_size)
    output_c = pos_in_row % (out_channels * batch_size) // batch_size
    output_b = output_idx % batch_size
    
    mask = (output_h < height_out) & (output_w < width_out) & (output_c < out_channels) & (output_b < batch_size)
    
    if tl.any(mask):
        # Load bias for this output channel
        bias = tl.load(bias_ptr + output_c, mask=output_c < out_channels)
        
        # Calculate input position for conv2d (stride=2)
        input_h = output_h * 2
        input_w = output_w * 2
        
        # Start with bias
        conv_result = bias.to(tl.float32)
        
        # Compute 3x3 conv2d at this position
        for kc in range(in_channels):
            for kh in range(3):
                for kw in range(3):
                    # Calculate input tensor position
                    input_ptr_idx = (
                        output_b * in_channels * height_in * width_in +
                        kc * height_in * width_in +
                        (input_h + kh) * width_in +
                        (input_w + kw)
                    )
                    
                    # Calculate weight position  
                    weight_ptr_idx = (
                        output_c * in_channels * 3 * 3 +
                        kc * 3 * 3 +
                        kh * 3 +
                        kw
                    )
                    
                    # Load input value (with bounds checking)
                    input_mask = (input_h + kh < height_in) and (input_w + kw < width_in)
                    input_val = tl.load(input_ptr + input_ptr_idx, 
                                      mask=input_mask and (kc < in_channels), 
                                      other=0.0)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + weight_ptr_idx,
                                       mask=(output_c < out_channels) and (kc < in_channels))
                    
                    # Accumulate result
                    conv_result += input_val.to(tl.float32) * weight_val.to(tl.float32)
        
        # Apply ReLU
        relu_result = tl.maximum(conv_result, 0.0)
        
        # Store final result
        output_ptr_idx = (
            output_b * out_channels * height_out * width_out +
            output_c * height_out * width_out +
            output_h * width_out + 
            output_w
        )
        tl.store(output_ptr + output_ptr_idx, relu_result.to(tl.float16), mask=mask)

# Kernel wrapper that launches the optimized kernel
@torch.fx.wrap
def simple_conv_relu(in_0, in_1, in_3):
    batch_size, in_channels, height_in, width_in = in_3.shape
    out_channels = in_0.shape[0]
    height_out = height_in // 2  # stride=2
    width_out = width_in // 2
    
    # Output tensor
    output = torch.empty((batch_size, out_channels, height_out, width_out), dtype=torch.float16, device=in_3.device)
    
    # Calculate grid size
    total_elements = batch_size * out_channels * height_out * width_out
    BLOCK_SIZE = 512
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv_relu_kernel[grid_size](
        in_3, in_1, in_0, output,
        batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out,
        BLOCK_SIZE
    )
    
    return output

# Replacement function - returns the optimized kernel
def replacement_func():
    return simple_conv_relu