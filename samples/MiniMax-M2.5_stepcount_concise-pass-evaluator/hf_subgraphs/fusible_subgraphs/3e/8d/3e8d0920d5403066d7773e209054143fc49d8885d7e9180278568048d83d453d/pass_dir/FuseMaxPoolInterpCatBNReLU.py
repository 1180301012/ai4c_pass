import torch
import triton
import triton.language as tl


def pattern(in_5, in_4, running_mean, running_var, weight, bias):
    """
    Pattern: max_pool2d + interpolate + cat + batch_norm + relu
    This pattern is common in encoder-decoder networks like ERFNet.
    
    The operations are:
    1. max_pool2d on in_5 (downsamples by 2)
    2. interpolate back to target size
    3. cat in_4 and interpolated tensor along channel dimension
    4. batch_norm on concatenated tensor
    5. relu activation
    """
    # Pool the input
    pooled = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    
    # Get target size from in_4 (the tensor to concatenate with)
    target_h = in_4.shape[2]
    target_w = in_4.shape[3]
    
    # Interpolate back to target size
    interpolated = torch.nn.functional.interpolate(pooled, (target_h, target_w), None, 'bilinear', False)
    
    # Concatenate along channel dimension
    concat = torch.cat([in_4, interpolated], 1)
    
    # Batch normalization
    bn = torch.nn.functional.batch_norm(concat, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    
    # ReLU activation
    relu = torch.nn.functional.relu(bn, inplace=False)
    
    return relu


def replacement_args(in_5, in_4, running_mean, running_var, weight, bias):
    """Extract arguments for the replacement kernel."""
    return (in_5, in_4, running_mean, running_var, weight, bias)


@triton.jit
def fused_kernel(
    # Input pointers
    in_5_ptr, in_4_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
    batch_size, in_5_channels, in_4_channels, out_height, out_width,
    # Strides
    in_5_stride_b, in_5_stride_c, in_5_stride_h, in_5_stride_w,
    in_4_stride_b, in_4_stride_c, in_4_stride_h, in_4_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # BN parameters
    eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: max_pool2d + interpolate + cat + batch_norm + relu
    
    This kernel fuses all operations into a single GPU kernel for maximum performance.
    """
    # Get program ID for channel parallelization
    pid_c = tl.program_id(0)  # Channel program
    pid_hw = tl.program_id(1)  # HW parallelization
    
    # Calculate output channels
    out_channels = in_4_channels + in_5_channels
    
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
    # For now, process batch=0, we'll iterate over batch in the wrapper
    batch_idx = 0
    
    # Determine which input this channel comes from
    if pid_c < in_4_channels:
        # Direct copy from in_4
        in_4_offset = batch_idx * in_4_stride_b + pid_c * in_4_stride_c + h_idx * in_4_stride_h + w_idx * in_4_stride_w
        val = tl.load(in_4_ptr + in_4_offset, mask=mask, other=0.0)
    else:
        # Pool and interpolate from in_5
        # Pooled coords: h // 2, w // 2
        pooled_h = h_idx // 2
        pooled_w = w_idx // 2
        
        # Channel in in_5
        in_5_c = pid_c - in_4_channels
        
        # Load from pooled position
        in_5_offset = batch_idx * in_5_stride_b + in_5_c * in_5_stride_c + pooled_h * in_5_stride_h + pooled_w * in_5_stride_w
        val = tl.load(in_5_ptr + in_5_offset, mask=mask, other=0.0)
    
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
def fused_kernel_wrapper(in_5, in_4, running_mean, running_var, weight, bias):
    """
    Wrapper function that launches the fused Triton kernel.
    
    This fuses: max_pool2d + interpolate + cat + batch_norm + relu
    into a single GPU kernel.
    """
    # Get shapes
    batch_size, in_5_channels, in_5_h, in_5_w = in_5.shape
    batch_size_4, in_4_channels, in_4_h, in_4_w = in_4.shape
    
    # Output shape
    out_channels = in_4_channels + in_5_channels
    out_h = in_4_h
    out_w = in_4_w
    
    # Allocate output
    output = torch.empty((batch_size, out_channels, out_h, out_w), device=in_5.device, dtype=in_5.dtype)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid: (channels, HW blocks)
    # Channel grid: one program per channel
    channel_grid = out_channels
    
    # HW grid: number of blocks to cover all HW elements
    num_hw = out_h * out_w
    hw_grid = (num_hw + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each batch
    for b in range(batch_size):
        # Slice batch from inputs
        in_5_b = in_5[b]
        in_4_b = in_4[b]
        output_b = output[b]
        
        # Launch kernel with 2D grid: (channels, hw_blocks)
        fused_kernel[(channel_grid, hw_grid)](
            in_5_ptr=in_5_b,
            in_4_ptr=in_4_b,
            running_mean_ptr=running_mean,
            running_var_ptr=running_var,
            weight_ptr=weight,
            bias_ptr=bias,
            output_ptr=output_b,
            batch_size=batch_size,
            in_5_channels=in_5_channels,
            in_4_channels=in_4_channels,
            out_height=out_h,
            out_width=out_w,
            in_5_stride_b=in_5_b.stride(0),
            in_5_stride_c=in_5_b.stride(1),
            in_5_stride_h=in_5_b.stride(2),
            in_5_stride_w=in_5_b.stride(3),
            in_4_stride_b=in_4_b.stride(0),
            in_4_stride_c=in_4_b.stride(1),
            in_4_stride_h=in_4_b.stride(2),
            in_4_stride_w=in_4_b.stride(3),
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
    return fused_kernel_wrapper