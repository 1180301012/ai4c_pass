import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, channel_shape, ln_weight, ln_bias, epsilon):
    return torch.nn.functional.layer_norm(x, channel_shape, ln_weight, ln_bias, epsilon)

# Argument extraction function
def replacement_args(x, channel_shape, ln_weight, ln_bias, epsilon):
    return (x, channel_shape, ln_weight, ln_bias, epsilon)

# The Triton kernel for optimized LayerNorm
@triton.jit
def layer_norm_kernel(
    x_ptr,  # Input tensor [batch, channels, height, width]
    out_ptr,  # Output tensor [batch, channels, height, width]
    weight_ptr,  # LayerNorm weight [channels]
    bias_ptr,  # LayerNorm bias [channels]
    batch_size,
    num_channels,
    height,
    width,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes one channel
    channel_id = tl.program_id(0)
    
    # Calculate the starting index for this channel in the input
    channel_start = channel_id * batch_size * height * width
    channel_end = (channel_id + 1) * batch_size * height * width
    
    # Calculate mean and variance for the channel
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(channel_start, channel_end):
        x_val = tl.load(x_ptr + i)
        sum_val += x_val
        sum_sq += x_val * x_val
    
    # Sequential mean/variance calculation
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(channel_start, channel_end):
        x_val = tl.load(x_ptr + i)
        sum_val += x_val
        sum_sq += x_val * x_val
    mean = sum_val / (batch_size * height * width)
    var = sum_sq / (batch_size * height * width) - mean * mean
    
    # Now, process elements of the channel in parallel
    block_idx = tl.program_id(1)
    start_idx = channel_start + block_idx * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, channel_end)
    
    for i in range(start_idx, end_idx):
        x_val = tl.load(x_ptr + i)
        normalized = (x_val - mean) / tl.sqrt(var + epsilon)
        out_val = normalized * tl.load(weight_ptr + channel_id) + tl.load(bias_ptr + channel_id)
        tl.store(out_ptr + i, out_val)

# Kernel wrapper
@torch.fx.wrap
def layer_norm_kernel_wrapper(x, channel_shape, weight, bias, epsilon):
    # channel_shape is like (N, 1, 1) -> N is the number of channels
    num_channels = channel_shape[0]
    
    # Get the actual input shape (batch, in_channels, height, width)
    batch_size, _, height, width = x.shape
    
    # Create output tensor with the same shape as input
    out = torch.empty_like(x)
    
    # Determine block size
    BLOCK_SIZE = 128
    
    # Calculate number of blocks per channel
    num_blocks_per_channel = (batch_size * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    layer_norm_kernel[(num_channels, num_blocks_per_channel)](
        x_ptr=x,
        out_ptr=out,
        weight_ptr=weight,
        bias_ptr=bias,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        epsilon=epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return layer_norm_kernel_wrapper