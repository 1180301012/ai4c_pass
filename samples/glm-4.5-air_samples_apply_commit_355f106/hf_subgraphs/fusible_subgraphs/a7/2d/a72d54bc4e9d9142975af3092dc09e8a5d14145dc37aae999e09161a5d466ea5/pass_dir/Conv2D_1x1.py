import torch
import triton
import triton.language as tl

# Pattern matching function - matches the conv2d operation
def pattern(in_6, in_0):
    return torch.conv2d(in_6, in_0, None, (1, 1), (1, 1), (1, 1), 1)

# Argument extraction function
def replacement_args(in_6, in_0):
    return (in_6, in_0)

# Optimized Conv2D kernel for 1x1 conv with 3x3 kernel
@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one batch and output channel combination
    batch_idx = pid // out_channels
    out_channel_idx = pid % out_channels
    
    if batch_idx >= batch_size or out_channel_idx >= out_channels:
        return
    
    # Process each input channel
    for c_in in range(in_channels):
        # Load 3x3 weights for this input channel and output channel
        kernel_idx = out_channel_idx * in_channels * 9 + c_in * 9
        kernel_weights = tl.load(
            weight_ptr + kernel_idx + tl.arange(0, 9),
            mask=tl.arange(0, 9) < 9,
            other=0.0
        )
        
        # Compute convolution for each spatial position
        for h in range(height):
            for w in range(width):
                # Handle padding with stride 1 and dilation 1
                spatial_sum = 0.0
                for kh in range(3):
                    for kw in range(3):
                        input_h = h + kh - 1  # padding of 1 on all sides
                        input_w = w + kw - 1
                        
                        if 0 <= input_h < height and 0 <= input_w < width:
                            # Load input value for this position and channel
                            input_ptr_loc = (input_ptr + 
                                           batch_idx * in_channels * height * width +
                                           c_in * height * width +
                                           input_h * width + input_w)
                            input_val = tl.load(input_ptr_loc)
                            
                            # Get corresponding kernel weight
                            kernel_weight = kernel_weights[kh * 3 + kw]
                            
                            # Accumulate
                            spatial_sum += input_val * kernel_weight
                
                # Store output
                output_idx = (batch_idx * out_channels * height * width +
                            out_channel_idx * height * width +
                            h * width + w)
                
                # Initialize output or accumulate
                current_output = tl.load(output_ptr + output_idx, other=0.0)
                tl.store(output_ptr + output_idx, current_output + spatial_sum)

# Simplified kernel wrapper focusing on the specific case
@torch.fx.wrap
def optimized_conv2d(input, weight):
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]
    
    # For 1x1 conv with 3x3 kernel, output is same spatial size
    output = torch.empty((N, C_out, H, W), dtype=input.dtype, device=input.device)
    
    # Each program handles one batch and output channel combination
    total_programs = N * C_out
    
    # Launch kernel with single dimension
    conv2d_kernel[(total_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=N,
        in_channels=C_in,
        out_channels=C_out,
        height=H,
        width=W,
        BLOCK_SIZE_M=1,  # Not used, keeping for compatibility
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_conv2d