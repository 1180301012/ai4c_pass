import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    batch_norm_out = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return batch_norm_out

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load parameters - simplified approach
    # We'll use the first dimension for channel indexing
    if tl.num_programs(axis=1) > 1:
        # Multi-program configuration
        channel_idx = tl.program_id(1)
        mean_val = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < tl.load(running_mean_ptr + 256), other=0.0)
        var_val = tl.load(running_var_ptr + channel_idx, mask=channel_idx < tl.load(running_var_ptr + 256), other=1.0)
        weight_val = tl.load(weight_ptr + channel_idx, mask=channel_idx < tl.load(weight_ptr + 256), other=1.0)
        bias_val = tl.load(bias_ptr + channel_idx, mask=channel_idx < tl.load(bias_ptr + 256), other=0.0)
    else:
        # Single program configuration - load first parameters
        mean_val = tl.load(running_mean_ptr + 0, mask=True, other=0.0)
        var_val = tl.load(running_var_ptr + 0, mask=True, other=1.0)
        weight_val = tl.load(weight_ptr + 0, mask=True, other=1.0)
        bias_val = tl.load(bias_ptr + 0, mask=True, other=0.0)
    
    # Broadcast parameters to match element count
    mean_val = tl.broadcast_to(mean_val, BLOCK_SIZE)
    var_val = tl.broadcast_to(var_val, BLOCK_SIZE)
    weight_val = tl.broadcast_to(weight_val, BLOCK_SIZE)
    bias_val = tl.broadcast_to(bias_val, BLOCK_SIZE)
    
    # Batch normalization computation
    var_inv = 1.0 / tl.sqrt(var_val + eps)
    output = (input_val - mean_val) * weight_val * var_inv + bias_val
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    # Get input shape - handle different tensor shapes
    if len(input_tensor.shape) == 4:  # NCHW format
        N, C, H, W = input_tensor.shape
        n_elements = N * C * H * W
        n_channels = C
    elif len(input_tensor.shape) == 2:  # Matrix format
        n_elements = input_tensor.shape[0] * input_tensor.shape[1]
        n_channels = input_tensor.shape[1]
    elif len(input_tensor.shape) == 1:  # Vector format
        n_elements = input_tensor.shape[0]
        n_channels = 1
    else:
        raise ValueError(f"Unsupported input shape: {input_tensor.shape}")
    
    # Output tensor
    output = torch.empty_like(input_tensor)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 1024
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with simpler grid configuration
    batch_norm_kernel[(n_blocks,)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=0.001,
        momentum=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_batch_norm