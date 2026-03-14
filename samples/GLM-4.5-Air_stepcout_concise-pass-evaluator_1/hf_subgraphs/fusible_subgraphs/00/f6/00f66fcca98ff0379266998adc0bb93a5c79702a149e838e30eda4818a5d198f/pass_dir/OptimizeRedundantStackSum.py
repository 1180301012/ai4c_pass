import torch
import triton
import triton.language as tl

def pattern(input_tensor, bias_tensor, weight_tensor, additional_input):
    # Match the Conv2D -> Stack -> Sum pattern
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv_result], dim=0)
    summed = stacked.sum(dim=0)
    concatenated = torch.cat([summed, additional_input], 1)
    
    # Return the final result and the intermediate convolution output
    # (this ensures the observable intermediate is available for the model's return)
    return summed, concatenated

def replacement_args(input_tensor, bias_tensor, weight_tensor, additional_input):
    return (input_tensor, bias_tensor, weight_tensor, additional_input)

@triton.jit
def optimized_conv_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, WARPS_PER_ROW: tl.constexpr
):
    # Each program computes a (BLOCK_SIZE_M, BLOCK_SIZE_N) block of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_offset = tl.program_id(2) * height * width
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Compute the range of channels this block should process
    k_offset = pid_n * BLOCK_SIZE_K
    
    # Load bias for the output channels
    bias_addr = weight_ptr + (pid_n * BLOCK_SIZE_K + k_offset) * 1 * 1
    bias = tl.load(bias_addr, mask=(k_offset < out_channels))
    bias = tl.broadcast(bias, (BLOCK_SIZE_M, 1))
    
    # Process channels
    max_k = tl.cdiv(in_channels, BLOCK_SIZE_K)
    for k in range(max_k):
        # Load input block
        input_addr = input_ptr + (batch_offset + pid_m * height + 0) * width * in_channels + k * BLOCK_SIZE_K
        input_block = tl.load(input_addr, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < in_channels))
        input_block = tl.reshape(input_block, (1, BLOCK_SIZE_K))
        
        # Load weight block
        weight_addr = weight_ptr + (pid_n * BLOCK_SIZE_K + k * BLOCK_SIZE_K) * 1 * 1
        if k < max_k - 1 or in_channels % BLOCK_SIZE_K == 0:
            weight_block = tl.load(weight_addr)
        else:
            mask = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < in_channels)
            weight_block = tl.load(weight_addr, mask=mask)
        
        # Convolution operation (1x1 kernel)
        weight_block = tl.reshape(weight_block, (BLOCK_SIZE_K, 1, 1))
        conv_result = input_block * weight_block
        
        # Sum over the spatial dimensions and accumulate
        conv_result = tl.sum(conv_result, [1, 2])
        accumulator += conv_result
    
    # Store result
    accumulator += bias
    output_addr = output_ptr + (batch_offset + pid_m * height + 0) * width + pid_n * BLOCK_SIZE_K
    tl.store(output_addr, accumulator)

@torch.fx.wrap
def optimized_conv(input_tensor, bias_tensor, weight_tensor, additional_input):
    # Determine shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Create output tensor (will be concat with additional_input later)
    conv_output = torch.empty((batch_size, out_channels, height, width), 
                            dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton kernel scheduling
    BLOCK_SIZE_M = 8  # Process 8 spatial elements at a time
    BLOCK_SIZE_N = 256  # Process 256 output channels at a time
    BLOCK_SIZE_K = 32  # Process 32 input channels at a time
    WARPS_PER_ROW = 4
    
    # Calculate grid dimensions
    num_m = height
    num_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    num_batch = batch_size
    
    # Launch kernel
    grid = (num_m, num_n, num_batch)
    
    optimized_conv_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=conv_output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        WARPS_PER_ROW=WARPS_PER_ROW
    )
    
    # Direct concatenation - no need for stack/sum overhead
    final_result = torch.cat([conv_output, additional_input], dim=1)
    
    return conv_output, final_result

def replacement_func():
    return optimized_conv