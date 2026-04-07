import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor):
    """Match 1x1 convolution with 1x1 stride and 1x1 padding"""
    result = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (1, 1), (1, 1), 1)
    return result

def replacement_args(input_tensor, weight_tensor):
    return (input_tensor, weight_tensor)

@triton.jit
def conv3x3_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    kernel_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Compute tile offsets for output
    pid_m = tl.program_id(0)  # batch and output channel
    pid_n = tl.program_id(1)  # spatial position
    
    # Create block masks
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offsets_m < batch_size * out_channels
    mask_n = offsets_n < in_height * in_width
    
    # Reshape offsets for matrix multiplication view
    out_channel_idx = offsets_m % out_channels
    batch_idx = offsets_m // out_channels
    
    # Load weight for this output channel
    weight_offsets = out_channel_idx[:, None] * (in_channels * kernel_size * kernel_size) + tl.arange(0, BLOCK_SIZE_K)
    weight_tile = tl.load(weight_ptr + weight_offsets, 
                        mask=mask_m[:, None] & (weight_offsets < out_channels * in_channels * kernel_size * kernel_size), 
                        other=0.0)
    
    # Reshape for better memory access
    weight_reshaped = weight_tile.reshape(out_channel_idx.shape[0], BLOCK_SIZE_K // (kernel_size * kernel_size), kernel_size, kernel_size)
    
    # Accumulate convolution result
    output_buffer = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input indices with padding
            h_base = (offsets_n % in_height) + kh
            w_base = (offsets_n // in_height) + kw
            
            # Pad with zeros if out of bounds
            mask_valid = (h_base < in_height) & (w_base < in_width)
            
            # Flatten input coordinates
            input_spatial = h_base * in_width + w_base
            
            # Add batch and channel dimensions
            input_offsets = batch_idx[:, None] * (in_height * in_width) + input_spatial[None, :] * in_channels + tl.arange(0, BLOCK_SIZE_K // (kernel_size * kernel_size))[None, None, :]
            
            # Load input tiles
            input_buffer = tl.load(input_ptr + input_offsets.to(tl.int64), 
                                 mask=mask_m[:, None, None] & mask_valid[:, None, None] & 
                                      (input_offsets < batch_size * in_height * in_width * in_channels), 
                                 other=0.0)
            
            # Extract corresponding weight slice and compute
            weight_slice = weight_reshaped[:, :, kh, kw]
            conv_result = tl.dot(input_buffer, weight_slice)
            output_buffer += conv_result
    
    # Store convolution result
    output_offsets = offsets_m[:, None] * (in_height * in_width) + offsets_n[None, :] 
    tl.store(output_ptr + output_offsets.to(tl.int64), output_buffer.to(tl.float32),
             mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_conv1x1(input_tensor, weight_tensor):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, kernel_h, kernel_w, _ = weight_tensor.shape
    
    # Get kernel size
    kernel_size = kernel_h
    
    # Compute output shape (for padding 1, stride 1, output size stays the same)
    output_height = in_height
    output_width = in_width
    output_shape = (batch_size, out_channels, output_height, output_width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Config block sizes based on tensor dimensions
    BLOCK_SIZE_M = 32  # Batch and output channels dimension  
    BLOCK_SIZE_N = 32  # Spatial dimension position
    BLOCK_SIZE_K = in_channels * kernel_size * kernel_size  # Kernel parameters
    
    # Grid configuration
    grid = (
        (batch_size * out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Create contiguous memory views for better memory access
    input_contiguous = input_tensor.contiguous()
    weight_contiguous = weight_tensor.contiguous()
    
    conv3x3_kernel[grid](
        input_contiguous,
        weight_contiguous,
        output,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_conv1x1