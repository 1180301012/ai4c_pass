import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, other_input):
    # Match the complete pattern: conv2d -> stack -> sum -> cat
    conv_result = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv_result], dim=0)
    summed = stacked.sum(dim=0)
    concatenated = torch.cat([summed, other_input], 1)
    return concatenated

def replacement_args(conv_input, weight, bias, other_input):
    return (conv_input, weight, bias, other_input)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, channels_out, channels_in, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple optimized conv2d for demonstration
    # In reality, this would be more complex with proper tiling
    pid = tl.program_id(0)
    
    if pid >= batch_size * channels_out:
        return
    
    # Calculate output position
    batch = pid // channels_out
    channel_out = pid % channels_out
    
    if batch >= batch_size or channel_out >= channels_out:
        return
    
    # Process spatial dimensions
    for h in range(height):
        for w in range(width):
            # Accumulate result for this position
            sum_val = 0.0
            
            # For 1x1 conv, we just multiply corresponding channels
            for c_in in range(channels_in):
                # Load input
                input_idx = batch * channels_in * height * width + c_in * height * width + h * width + w
                val = tl.load(input_ptr + input_idx, other=0.0)
                
                # Load weight
                weight_idx = channel_out * channels_in + c_in
                weight_val = tl.load(weight_ptr + weight_idx, other=0.0)
                
                sum_val += val * weight_val
            
            # Add bias
            bias_val = tl.load(bias_ptr + channel_out, other=0.0)
            sum_val += bias_val
            
            # Store result
            output_idx = batch * channels_out * height * width + channel_out * height * width + h * width + w
            tl.store(output_ptr + output_idx, sum_val)

@torch.fx.wrap
def optimized_conv2d(input_tensor, weight, bias):
    # Get tensor shapes
    batch_size, channels_in, height, width = input_tensor.shape
    channels_out = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, channels_out, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch Triton kernel
    total_elements = batch_size * channels_out
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    optimized_conv2d_kernel[grid](
        input_tensor, weight, bias, output,
        batch_size, channels_out, channels_in, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, channels_out, channels_in, height, width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Optimized 1x1 conv2d kernel with proper tiling
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute block ranges
    start_m = m * BLOCK_SIZE_M
    start_n = n * BLOCK_SIZE_N
    end_m = min(start_m + BLOCK_SIZE_M, batch_size * channels_out)
    end_n = min(start_n + BLOCK_SIZE_N, height * width)
    
    if start_m >= batch_size * channels_out or start_n >= height * width:
        return
    
    # Create shared memory for weights in this block
    shared_weight = tl.zeros((BLOCK_SIZE_M, channels_in), dtype=tl.float32)
    shared_bias = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Load weights and bias into shared memory
    for idx in range(tl.cdiv(channels_out, BLOCK_SIZE_M)):
        weight_m = start_m + idx * BLOCK_SIZE_M
        if weight_m < channels_out:
            for c_in in range(channels_in):
                weight_idx = weight_m * channels_in + c_in
                shared_weight[idx * BLOCK_SIZE_M + (weight_m - start_m), c_in] = tl.load(weight_ptr + weight_idx, other=0.0)
            bias_val = tl.load(bias_ptr + weight_m, other=0.0)
            shared_bias[idx * BLOCK_SIZE_M + (weight_m - start_m)] = bias_val
    
    # Process each spatial position in the block
    for hw in range(start_n, end_n):
        h = hw // width
        w = hw % width
        
        # Process all outputs in this block
        for idx in range(tl.cdiv(channels_out, BLOCK_SIZE_M)):
            weight_m = start_m + idx * BLOCK_SIZE_M
            if weight_m < channels_out:
                batch_idx = weight_m // channels_out
                channel_out_idx = weight_m % channels_out
                
                # Load input for this position
                input_val = tl.zeros((channels_in,), dtype=tl.float32)
                for c_in in range(channels_in):
                    input_idx = batch_idx * channels_in * height * width + c_in * height * width + h * width + w
                    input_val[c_in] = tl.load(input_ptr + input_idx, other=0.0)
                
                # Compute convolution result
                result = 0.0
                for c_in in range(channels_in):
                    result += shared_weight[idx * BLOCK_SIZE_M + (weight_m - start_m), c_in] * input_val[c_in]
                result += shared_bias[idx * BLOCK_SIZE_M + (weight_m - start_m)]
                
                # Store result
                output_idx = batch_idx * channels_out * height * width + channel_out_idx * height * width + h * width + w
                tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def optimized_conv(cat_input, weight, bias, other_input):
    # Get tensor shapes
    batch_size, channels_in, height, width = cat_input.shape
    channels_out = weight.shape[0]
    
    # Perform optimized conv2d
    conv_result = torch.empty((batch_size, channels_out, height, width), 
                             dtype=cat_input.dtype, device=cat_input.device)
    
    # Launch Triton kernel with optimized tiling
    BLOCK_SIZE_M = 64  # Output channels per block
    BLOCK_SIZE_N = 1024  # Spatial positions per block
    
    grid_m = triton.cdiv(batch_size * channels_out, BLOCK_SIZE_M)
    grid_n = triton.cdiv(height * width, BLOCK_SIZE_N)
    
    optimized_conv2d_kernel[(grid_m, grid_n)](
        cat_input, weight, bias, conv_result,
        batch_size, channels_out, channels_in, height, width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Direct concatenation without redundant stack-sum operations
    return torch.cat([conv_result, other_input], 1)

def replacement_func():
    return optimized_conv