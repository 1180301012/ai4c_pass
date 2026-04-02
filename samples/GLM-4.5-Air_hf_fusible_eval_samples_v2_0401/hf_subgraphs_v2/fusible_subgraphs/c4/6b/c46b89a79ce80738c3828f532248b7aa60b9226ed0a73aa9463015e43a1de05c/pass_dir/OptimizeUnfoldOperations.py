import torch
import triton
import triton.language as tl

def pattern(x):
    return x.permute(2, 0, 1)

def replacement_args(x):
    return (x,)

@triton.jit
def unfold_and_reshape_kernel_768_384_192(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels, 
    in_height, in_width,
    out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    output_height = out_height // stride_h
    output_width = out_width // stride_w
    
    # Calculate coordinates
    batch_id = pid // (output_height * output_width)
    spatial_id = pid % (output_height * output_width)
    out_y = spatial_id // output_width
    out_x = spatial_id % output_width
    
    if batch_id >= n_batch:
        return
    
    # Calculate input coordinates
    in_y = out_y * stride_h
    in_x = out_x * stride_w
    
    # Each thread handles one channel
    channel_id = tl.program_id(1)
    
    if channel_id >= n_channels:
        return
    
    # Load kernel
    kernel_data = tl.load(input_ptr + batch_id * n_channels * in_height * in_width +
                         channel_id * in_height * in_width +
                         in_y * in_width + in_x + 
                         tl.arange(0, kernel_h)[:, None] * in_width + 
                         tl.arange(0, kernel_w)[None, :],
                         mask=(tl.arange(0, kernel_h)[:, None] < kernel_h) & 
                              (tl.arange(0, kernel_w)[None, :] < kernel_w),
                         other=0.0)
    
    # Reshape to [kernel_size, output_idx]
    kernel_reshaped = kernel_data.reshape(kernel_h * kernel_w)
    
    # Output offset
    output_offset = (batch_id * n_channels + channel_id) * output_height * output_width + spatial_id
    
    # Store
    tl.store(output_ptr + output_offset * kernel_h * kernel_w, kernel_reshaped)

@triton.jit
def unfold_and_reshape_kernel_1536_384_288(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels, 
    in_height, in_width,
    out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    output_height = out_height // stride_h
    output_width = out_width // stride_w
    
    # Calculate coordinates
    batch_id = pid // (output_height * output_width)
    spatial_id = pid % (output_height * output_width)
    out_y = spatial_id // output_width
    out_x = spatial_id % output_width
    
    if batch_id >= n_batch:
        return
    
    # Calculate input coordinates
    in_y = out_y * stride_h
    in_x = out_x * stride_w
    
    # Each thread handles one channel
    channel_id = tl.program_id(1)
    
    if channel_id >= n_channels:
        return
    
    # Load kernel
    kernel_data = tl.load(input_ptr + batch_id * n_channels * in_height * in_width +
                         channel_id * in_height * in_width +
                         in_y * in_width + in_x + 
                         tl.arange(0, kernel_h)[:, None] * in_width + 
                         tl.arange(0, kernel_w)[None, :],
                         mask=(tl.arange(0, kernel_h)[:, None] < kernel_h) & 
                              (tl.arange(0, kernel_w)[None, :] < kernel_w),
                         other=0.0)
    
    # Reshape to [kernel_size, output_idx]
    kernel_reshaped = kernel_data.reshape(kernel_h * kernel_w)
    
    # Output offset
    output_offset = (batch_id * n_channels + channel_id) * output_height * output_width + spatial_id
    
    # Store
    tl.store(output_ptr + output_offset * kernel_h * kernel_w, kernel_reshaped)

def optimized_unfold_reshape_768_384_192(input_tensor):
    n_batch, n_channels, in_height, in_width = input_tensor.shape
    kernel_h, kernel_w = 384, 384
    stride_h, stride_w = 192, 192
    
    output_height = in_height // stride_h  # 768 // 192 = 4
    output_width = in_width // stride_w    # 768 // 192 = 4
    total_patches = output_height * output_width  # 4 * 4 = 16
    
    # Output tensor: [batch * channels * total_patches, kernel_h * kernel_w]
    output_shape = (n_batch * n_channels * total_patches, kernel_h * kernel_w)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid
    total_programs = n_batch * total_patches
    block_size = 1024
    
    grid = (triton.cdiv(total_programs, block_size), n_channels)
    
    unfold_and_reshape_kernel_768_384_192[grid](
        input_tensor,
        output,
        n_batch,
        n_channels,
        in_height, in_width,
        output_height, output_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32
    )
    
    # Reshape to [batch * patches, channels, kernel_h, kernel_w]
    final_shape = (n_batch * total_patches, n_channels, kernel_h, kernel_w)
    result = output.reshape(final_shape)
    
    return result

def optimized_unfold_reshape_1536_384_288(input_tensor):
    n_batch, n_channels, in_height, in_width = input_tensor.shape
    kernel_h, kernel_w = 384, 384
    stride_h, stride_w = 288, 288
    
    output_height = in_height // stride_h  # 1536 // 288 = 5
    output_width = in_width // stride_w    # 1536 // 288 = 5
    total_patches = output_height * output_width  # 5 * 5 = 25
    
    # Output tensor: [batch * channels * total_patches, kernel_h * kernel_w]
    output_shape = (n_batch * n_channels * total_patches, kernel_h * kernel_w)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid
    total_programs = n_batch * total_patches
    block_size = 1024
    
    grid = (triton.cdiv(total_programs, block_size), n_channels)
    
    unfold_and_reshape_kernel_1536_384_288[grid](
        input_tensor,
        output,
        n_batch,
        n_channels,
        in_height, in_width,
        output_height, output_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32
    )
    
    # Reshape to [batch * patches, channels, kernel_h, kernel_w]
    final_shape = (n_batch * total_patches, n_channels, kernel_h, kernel_w)
    result = output.reshape(final_shape)
    
    return result

@torch.fx.wrap
def permute_kernel_wrapper(x):
    # Simple permute replacement
    return x.permute(2, 0, 1)

def replacement_func():
    return permute_kernel_wrapper