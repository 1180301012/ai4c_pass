import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    x_shape,
    channel_dim,
    epsilon: tl.constexpr = 1e-05,
    BLOCK_SIZE: tl.constexpr = 1024
):
    # Each thread handles one channel
    channel = tl.program_id(0)
    # Initialize accumulator for spatial dimensions
    x = tl.zeros((x_shape[1],), dtype=tl.float32)
    
    # Load all spatial values for this channel
    for i in range(x_shape[1]):
        x[i] = tl.load(x_ptr + channel * x_shape[1] + i)
    
    # Compute mean and variance
    mean = tl.sum(x) / x_shape[1]
    var = tl.sum((x - mean)**2) / x_shape[1]
    std = tl.sqrt(var + epsilon)
    
    # Normalize and apply scale/shift
    normalized = (x - mean) / std
    normalized = normalized * tl.load(weight_ptr + channel) + tl.load(bias_ptr + channel)
    
    # Store result
    tl.store(out_ptr + channel * x_shape[1], normalized)

@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, epsilon=1e-05):
    x_shape = x.shape
    channel_dim = x_shape[-1]
    
    # Allocate output tensor of the same shape as x
    out = torch.empty_like(x)
    
    # Configure kernel grid
    grid = (1,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        x_shape=x_shape,
        channel_dim=channel_dim,
        epsilon=epsilon,
        BLOCK_SIZE=1024
    )
    
    return out
def replacement_func():
    return layer_norm_wrapper