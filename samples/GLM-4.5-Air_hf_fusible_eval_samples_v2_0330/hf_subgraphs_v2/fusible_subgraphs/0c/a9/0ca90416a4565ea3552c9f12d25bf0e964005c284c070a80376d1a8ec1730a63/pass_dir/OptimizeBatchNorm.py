import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    """
    Pattern for batch normalization operation.
    Matches: torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    """
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return tmp_5

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    """
    Extract arguments for the optimized batch normalization kernel.
    Returns: input tensor, running mean, running var, weight, bias, momentum, eps
    """
    return (in_4, in_0, in_1, in_3, in_2, 0.1, 0.001)

@triton.jit
def optimized_batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    num_channels,
    height,
    width,
    momentum,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for batch normalization.
    Applies: y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    # Each program handles one channel
    pid = tl.program_id(0)
    
    if pid >= num_channels:
        return
    
    # Initialize accumulators for channel-wide reduction
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Calculate channel stride
    channel_stride = height * width
    
    # Thread-local reduction over batch, height, width
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                # Calculate global index
                idx = ((b * num_channels + pid) * height + h) * width + w
                x = tl.load(x_ptr + idx, other=0.0)
                
                # Accumulate sums
                sum_x += float(x)
                sum_x2 += float(x) * float(x)
    
    # Calculate mean and variance for this channel
    channel_mean = sum_x / (batch_size * height * width)
    channel_var = sum_x2 / (batch_size * height * width) - channel_mean * channel_mean
    
    # Load normalization parameters
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    weight = tl.load(weight_ptr + pid) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + pid) if bias_ptr is not None else 0.0
    
    # Calculate running mean and variance (for inference, we use the stored statistics)
    # In training, we would update these, but for inference optimization, we use existing
    
    # Apply normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv_std * weight + bias
    
    # Store result
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                idx = ((b * num_channels + pid) * height + h) * width + w
                x = tl.load(x_ptr + idx, other=0.0)
                
                # Compute output
                y_val = (float(x) - float(mean)) * float(inv_std) * float(weight) + float(bias)
                tl.store(output_ptr + idx, y_val)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias, momentum=0.1, eps=0.001):
    """
    Optimized batch normalization function using Triton.
    Uses pre-computed running statistics for inference.
    """
    # Get tensor dimensions
    batch_size, num_channels, height, width = input_tensor.shape
    
    # Create output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Calculate grid size (one program per channel)
    grid_size = (num_channels,)
    
    # Launch kernel
    optimized_batch_norm_kernel[grid_size](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output_tensor,
        batch_size,
        num_channels,
        height,
        width,
        momentum,
        eps,
        1024,  # BLOCK_SIZE
    )
    
    return output_tensor

def replacement_func():
    """
    Returns the optimized batch normalization function.
    """
    return optimized_batch_norm