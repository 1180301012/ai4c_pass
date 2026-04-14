import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_2, in_3, in_4):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_6, tmp_7, tmp_8

def replacement_args(in_5, in_1, in_2, in_3, in_4):
    return (in_5, in_1, in_2, in_3, in_4)

@triton.jit
def fused_pool_bn_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels: tl.constexpr,
    n_batches: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program handles one channel
    pid = tl.program_id(0)
    if pid >= n_channels:
        return
    
    # Load normalization parameters for this channel
    running_mean = tl.load(running_mean_ptr + pid)
    running_var = tl.load(running_var_ptr + pid)
    weight = tl.load(weight_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    
    # Calculate standard deviation
    std = tl.sqrt(running_var + eps)
    
    # Process all batches for this channel
    for batch_idx in range(n_batches):
        # Compute the spatial average for this batch and channel
        spatial_sum = 0.0
        spatial_count = 0
        
        # Load and sum all spatial elements
        for h in range(height):
            for w in range(width):
                input_offset = (batch_idx * n_channels + pid) * height * width + h * width + w
                val = tl.load(input_ptr + input_offset)
                spatial_sum += val
                spatial_count += 1
        
        # Compute spatial average
        spatial_avg = spatial_sum / spatial_count
        
        # Apply batch normalization: y = (x - mean) / std * weight + bias
        norm_val = (spatial_avg - running_mean) / std
        bn_val = norm_val * weight + bias
        
        # Apply ReLU activation
        relu_val = tl.maximum(bn_val, 0.0)
        
        # Store the result at [batch_idx, pid, 0, 0]
        output_offset = (batch_idx * n_channels + pid) * 1 * 1 + 0 * 1 + 0
        tl.store(output_ptr + output_offset, relu_val)

@torch.fx.wrap
def fused_pool_bn_relu(input, running_mean, running_var, weight, bias):
    # Get input dimensions: [n_batches, n_channels, 8, 8]
    n_batches, n_channels, height, width = input.shape
    
    # Output will be [n_batches, n_channels, 1, 1]
    output = torch.empty((n_batches, n_channels, 1, 1), dtype=input.dtype, device=input.device)
    
    # Launch kernel with original tensor shapes
    fused_pool_bn_relu_kernel[(n_channels,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_channels=n_channels,
        n_batches=n_batches,
        eps=1e-05,
        BLOCK_SIZE=1024,
        height=height,
        width=width
    )
    
    return output

def replacement_func():
    return fused_pool_bn_relu