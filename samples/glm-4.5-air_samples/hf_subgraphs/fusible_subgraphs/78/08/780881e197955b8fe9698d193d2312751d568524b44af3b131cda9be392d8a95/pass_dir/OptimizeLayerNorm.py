import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    # Pattern matches layer norm operation
    normalized = torch.nn.functional.layer_norm(input, (96,), weight, bias, 1e-05)
    return normalized

def replacement_args(input, weight, bias):
    # Extract arguments for the optimized kernel
    return (input, weight, bias)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,  # sequence length
    C: tl.constexpr,  # number of channels must be constexpr
    eps: tl.constexpr,
):
    # Each program handles one position and one channel
    pos_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Calculate global offset for this position and channel
    offset = pos_id * C + channel_id
    
    # Boundary check
    mask = (pos_id < N) & (channel_id < C)
    
    # Load input data, weight and bias for this position and channel
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    weight_val = tl.load(weight_ptr + channel_id, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + channel_id, mask=mask, other=0.0)
    
    # Load all elements for this position across all channels to compute mean/variance
    pos_offset = pos_id * C
    pos_data = tl.load(input_ptr + pos_offset + tl.arange(0, C), mask=(pos_id < N))
    
    # Calculate mean and variance across channels for this position
    x_mean = tl.sum(pos_data) / C
    x_var = tl.sum((pos_data - x_mean) * (pos_data - x_mean)) / C
    x_inv_std = tl.rsqrt(x_var + eps)
    
    # Apply layer normalization for this channel
    x_normalized = (x - x_mean) * x_inv_std
    output = x_normalized * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + offset, output, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(input, weight, bias):
    # Handle different input shapes (1, N, C)
    input_reshaped = input.reshape(-1, weight.shape[0])  # Reshape to (N, C)
    N, C = input_reshaped.shape
    eps = 1e-05
    
    # Use 2D grid where each program handles one position and one channel
    grid = (N, C, 1)
    
    output = torch.empty_like(input_reshaped)
    
    layer_norm_kernel[grid](
        input_ptr=input_reshaped,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        C=C,
        eps=eps,
    )
    
    return output.reshape_as(input)  # Reshape back to original shape

def replacement_func():
    return optimized_layer_norm