import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_mean_kernel(
    input_ptr,
    mean_output_ptr,
    activated_output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    keepdim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes SiLU activation and mean over spatial dimensions in a single pass.
    """
    # Each program handles one channel within one batch
    program_id = tl.program_id(0)
    
    batch_idx = program_id // channels
    channel_idx = program_id % channels
    
    # Calculate the starting offset for this batch and channel
    offset = batch_idx * channels * height * width + channel_idx * height * width
    
    # Initialize accumulators for sum and count
    sum_val = 0.0
    count = height * width
    
    # Process the spatial dimensions
    for h in range(height):
        for w in range(width):
            idx = offset + h * width + w
            val = tl.load(input_ptr + idx)
            
            # Compute SiLU: x * sigmoid(x)
            sigmoid_val = 1.0 / (1.0 + tl.exp(-val))
            silu_val = val * sigmoid_val
            
            # Accumulate for mean
            sum_val += silu_val
            
            # Store activated value if needed (activated_output_ptr is not None)
            if activated_output_ptr is not None:
                store_idx = idx
                tl.store(activated_output_ptr + store_idx, silu_val)
    
    # Compute mean
    mean_val = sum_val / tl.constexpr(count)
    
    # Store mean result
    if keepdim:
        # For keepdim=True, output shape is [batch, channels, 1, 1]
        mean_offset = batch_idx * channels * 1 * 1 + channel_idx * 1 * 1
        tl.store(mean_output_ptr + mean_offset, mean_val)
    else:
        # For keepdim=False, output shape is [batch, channels]
        mean_offset = batch_idx * channels + channel_idx
        tl.store(mean_output_ptr + mean_offset, mean_val)


def silu_and_mean_triton(x, keepdim=False, return_order=0):
    """
    Fused SiLU + Mean computation using Triton.
    
    Args:
        x: Input tensor of shape [batch, channels, height, width]
        keepdim: Whether to keep dimensions for mean output
        return_order: 0 means return (activated, mean), 1 means return (mean, activated)
    """
    batch_size, channels, height, width = x.shape
    
    # Create output tensors
    if keepdim:
        mean_shape = (batch_size, channels, 1, 1)
    else:
        mean_shape = (batch_size, channels)
    
    mean_output = torch.empty(mean_shape, device=x.device, dtype=x.dtype)
    activated_output = torch.empty_like(x)
    
    # Calculate grid
    num_programs = batch_size * channels
    
    # Launch kernel - use the wrapped version
    return silu_and_mean_kernel_wrapper(
        x, mean_output, activated_output,
        batch_size, channels, height, width, keepdim, num_programs)


@torch.fx.wrap
def silu_and_mean_kernel_wrapper(input_tensor, mean_output, activated_output,
                                  batch_size, channels, height, width, keepdim, num_programs):
    """Wrapper for Triton kernel to prevent dynamo tracing."""
    silu_and_mean_kernel[(num_programs,)](
        input_tensor,
        mean_output,
        activated_output,
        batch_size,
        channels,
        height,
        width,
        keepdim,
        BLOCK_SIZE=1024,
    )
    return mean_output, activated_output


# Pattern matching function - matches keepdim=False pattern
# The pattern only matches the mean operation - the silu is assumed to be applied before
# return (mean, activated) - order is tmp_1, tmp_0
def pattern(x):
    tmp_1 = x.mean((2, 3))
    return tmp_1, x


def replacement_args(x):
    return (x,)


def replacement_func():
    # return_order=1 means (mean, activated)
    def wrapper(x):
        return silu_and_mean_triton(x, keepdim=False, return_order=1)
    return wrapper


# Pattern matching function - matches keepdim=True pattern
# The pattern only matches the mean operation - the silu is assumed to be applied before
# return (activated, mean) - order is tmp_0, tmp_1
def pattern_keepdim(x):
    tmp_1 = x.mean((2, 3), keepdim=True)
    return x, tmp_1


def replacement_args_keepdim(x):
    return (x,)


def replacement_func_keepdim():
    # return_order=0 means (activated, mean)
    def wrapper(x):
        return silu_and_mean_triton(x, keepdim=True, return_order=0)
    return wrapper