import torch
import triton
import triton.language as tl


def pattern(in_0, running_mean, running_var, bias, weight, in_5):
    """
    Pattern: max_pool2d + interpolate + cat + batch_norm + relu
    This is variant where max_pool2d is applied to the input image (in_0).
    
    The operations are:
    1. max_pool2d on in_0 (downsamples by 2)
    2. interpolate back to target size (match in_5 size)
    3. cat in_5 and interpolated tensor along channel dimension
    4. batch_norm on concatenated tensor
    5. relu activation
    """
    # Pool the input image
    pooled = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    
    # Get target size from in_5 (the tensor to concatenate with)
    target_h = in_5.shape[2]
    target_w = in_5.shape[3]
    
    # Interpolate back to target size
    interpolated = torch.nn.functional.interpolate(pooled, (target_h, target_w), None, 'bilinear', False)
    
    # Concatenate along channel dimension
    concat = torch.cat([in_5, interpolated], 1)
    
    # Batch normalization
    bn = torch.nn.functional.batch_norm(concat, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    
    # ReLU activation
    relu = torch.nn.functional.relu(bn, inplace=False)
    
    return relu


def replacement_args(in_0, running_mean, running_var, bias, weight, in_5):
    """Extract arguments for the replacement kernel."""
    return (in_0, in_5, running_mean, running_var, weight, bias)


@triton.jit
def fused_kernel_v2(
    # Input pointers
    in_0_ptr, in_5_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
    in_0_channels, in_5_channels, out_height, out_width,
    # Strides
    in_0_stride_b, in_0_stride_c, in_0_stride_h, in_0_stride_w,
    in_5_stride_b, in_5_stride_c, in_5_stride_h, in_5_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # BN parameters
    eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel variant 2: max_pool2d + interpolate + cat + batch_norm + relu
    Where max_pool is applied to the input image (in_0).
    """
    # Get program ID for channel parallelization
    pid_c = tl.program_id(0)  # Channel program
    pid_hw = tl.program_id(1)  # HW parallelization
    
    # Calculate output channels
    out_channels = in_5_channels + in_0_channels
    
    # Only process valid channels
    if pid_c >= out_channels:
        return
    
    # Calculate HW offset
    hw_offset = pid_hw * BLOCK_SIZE
    hw_offsets = hw_offset + tl.arange(0, BLOCK_SIZE)
    
    # Total HW elements
    num_hw = out_height * out_width
    mask = hw_offsets < num_hw
    
    # Calculate h, w indices
    h_idx = hw_offsets // out_width
    w_idx = hw_offsets % out_width
    
    # Calculate batch index (assuming batch is the first dimension)
    batch_idx = 0
    
    # Determine which input this channel comes from
    if pid_c < in_5_channels:
        # Direct copy from in_5 (conv_out)
        in_5_offset = batch_idx * in_5_stride_b + pid_c * in_5_stride_c + h_idx * in_5_stride_h + w_idx * in_5_stride_w
        val = tl.load(in_5_ptr + in_5_offset, mask=mask, other=0.0)
    else:
        # Pool and interpolate from in_0 (input image)
        # Pooled coords: h // 2, w // 2
        pooled_h = h_idx // 2
        pooled_w = w_idx // 2
        
        # Channel in in_0
        in_0_c = pid_c - in_5_channels
        
        # Load from pooled position
        in_0_offset = batch_idx * in_0_stride_b + in_0_c * in_0_stride_c + pooled_h * in_0_stride_h + pooled_w * in_0_stride_w
        val = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0)
    
    # Load BN parameters
    mean = tl.load(running_mean_ptr + pid_c)
    var = tl.load(running_var_ptr + pid_c)
    w = tl.load(weight_ptr + pid_c)
    b = tl.load(bias_ptr + pid_c)
    
    # Batch_norm: (x - mean) / sqrt(var + eps) * weight + bias
    var_eps = var + eps
    std = tl.sqrt(var_eps)
    val = (val - mean) / std * w + b
    
    # ReLU
    val = tl.where(val > 0, val, 0.0)
    
    # Store to output
    out_offset = batch_idx * out_stride_b + pid_c * out_stride_c + h_idx * out_stride_h + w_idx * out_stride_w
    tl.store(output_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper_v2(in_0, in_5, running_mean, running_var, weight, bias):
    """
    Wrapper function that launches the fused Triton kernel (variant 2).
    
    This fuses: max_pool2d + interpolate + cat + batch_norm + relu
    where max_pool is applied to the input image (in_0).
    """
    # Get shapes
    batch_size, in_0_channels, in_0_h, in_0_w = in_0.shape
    batch_size_5, in_5_channels, in_5_h, in_5_w = in_5.shape
    
    # Output shape
    out_channels = in_5_channels + in_0_channels
    out_h = in_5_h
    out_w = in_5_w
    
    # Allocate output
    output = torch.empty((batch_size, out_channels, out_h, out_w), device=in_0.device, dtype=in_0.dtype)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid: (channels, HW blocks)
    channel_grid = out_channels
    num_hw = out_h * out_w
    hw_grid = (num_hw + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each batch
    for b in range(batch_size):
        # Slice batch from inputs
        in_0_b = in_0[b]
        in_5_b = in_5[b]
        output_b = output[b]
        
        # Launch kernel with 2D grid: (channels, hw_blocks)
        fused_kernel_v2[(channel_grid, hw_grid)](
            in_0_ptr=in_0_b,
            in_5_ptr=in_5_b,
            running_mean_ptr=running_mean,
            running_var_ptr=running_var,
            weight_ptr=weight,
            bias_ptr=bias,
            output_ptr=output_b,
            in_0_channels=in_0_channels,
            in_5_channels=in_5_channels,
            out_height=out_h,
            out_width=out_w,
            in_0_stride_b=in_0_b.stride(0),
            in_0_stride_c=in_0_b.stride(1),
            in_0_stride_h=in_0_b.stride(2),
            in_0_stride_w=in_0_b.stride(3),
            in_5_stride_b=in_5_b.stride(0),
            in_5_stride_c=in_5_b.stride(1),
            in_5_stride_h=in_5_b.stride(2),
            in_5_stride_w=in_5_b.stride(3),
            out_stride_b=output_b.stride(0),
            out_stride_c=output_b.stride(1),
            out_stride_h=output_b.stride(2),
            out_stride_w=output_b.stride(3),
            eps=0.001,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output


def replacement_func():
    """Return the replacement function."""
    return fused_kernel_wrapper_v2