import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, tensor_to_concat):
    # Match the entire computation sequence:
    # conv2d -> stack -> sum -> cat (with redundant stack+sum)
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv_result], dim=0)
    summed = stacked.sum(dim=0)
    final_result = torch.cat([summed, tensor_to_concat], 1)
    return final_result

def replacement_args(input_tensor, weight_tensor, bias_tensor, tensor_to_concat):
    return (input_tensor, weight_tensor, bias_tensor, tensor_to_concat)

@triton.jit
def fused_conv_cat_kernel(
    x_ptr, weight_ptr, bias_ptr, concat_ptr, out_ptr,
    batch_size, in_channels, out_channels, 
    height, width, concat_channels,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Fused Conv2D + Concat kernel that skips redundant stack+sum operations"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(out_channels + concat_channels, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(height * width, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    # Compute pointers for the expanded channel dimension
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = offs_m[:out_channels + concat_channels]
    offs_n = offs_n[:height * width]
    
    # Create masks for bounds checking
    mask_m = offs_m < (out_channels + concat_channels)
    mask_n = offs_n < height * width
    
    # Initialize output tensor
    out = tl.zeros((height, width, out_channels + concat_channels), dtype=tl.float32)
    
    # Process conv2d part (first out_channels)
    conv_part = tl.zeros((height, width, out_channels), dtype=tl.float32)
    
    # Simplified 1x1 conv implementation
    for b in range(batch_size):
        # Load input data for this batch
        x_base = x_ptr + b * in_channels * height * width
        x_vals = tl.load(x_base + offs_n[:, None] * in_channels, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Load bias
        bias_vals = tl.load(bias_ptr + offs_m[:out_channels], mask=offs_m[:out_channels] < out_channels, other=0.0).to(tl.float32)
        
        # Load weights and compute convolution (simplified for 1x1)
        for c_out in range(0, out_channels, 8):  # Process in chunks
            chunk_size = min(8, out_channels - c_out)
            if c_out < out_channels:
                weight_vals = tl.load(weight_ptr + (offs_m[:chunk_size] * in_channels) + c_out * in_channels, 
                                    mask=offs_m[:chunk_size] < out_channels, other=0.0).to(tl.float32)
                # Simplified matrix multiplication
                conv_contrib = (x_vals * weight_vals[:, None]).sum(axis=-1)
                conv_part += conv_contrib
        
        # Add bias
        conv_part += bias_vals[None, None, :]
        
        # Store conv result 
        store_base = out_ptr + b * (out_channels + concat_channels) * height * width
        conv_part_mask = (offs_m[:, None] * height + (offs_n // width)[:, None]) * width + (offs_n % width)
        conv_part_mask = (offs_m < out_channels)[:, None] & mask_n[:, None]
        
        out_slice = conv_part
        tl.store(store_base + conv_part_mask, out_slice, mask=conv_part_mask)
    
    # Process concatenation part (concat_channels)
    concat_base = concat_ptr + batch_size * concat_channels * height * width
    concat_vals = tl.load(concat_base + offs_n[:, None] * concat_channels, 
                         mask=offs_n[:, None] < height * width, other=0.0).to(tl.float32)
    
    # Store concatenation part 
    concat_mask = (offs_m >= out_channels)[:, None] & mask_n[:, None]
    store_base = out_ptr + batch_size * (out_channels + concat_channels) * height * width
    tl.store(store_base + ((offs_m - out_channels)[:, None] * height + (offs_n // width)[:, None]) * width + (offs_n % width), 
             concat_vals, mask=concat_mask)

@torch.fx.wrap
def optimized_fused_conv_cat(input_tensor, weight_tensor, bias_tensor, tensor_to_concat):
    """Optimized Conv2D + Concat that skips redundant operations using Triton"""
    
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight_tensor.shape
    concat_channels = tensor_to_concat.shape[1]
    
    # Determine output shape
    output_channels = out_channels + concat_channels
    output_shape = (batch_size, output_channels, height, width)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    
    # Calculate grid size
    total_output_channels = output_channels
    grid_m = (total_output_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = grid_m * grid_n
    
    # Launch kernel
    fused_conv_cat_kernel[grid_size](
        x_ptr=input_tensor,
        weight_ptr=weight_tensor, 
        bias_ptr=bias_tensor,
        concat_ptr=tensor_to_concat,
        out_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        concat_channels=concat_channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return optimized_fused_conv_cat