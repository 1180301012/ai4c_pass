import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor, weight, bias, normalized_shape, eps):
    return torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps)

def replacement_args(input_tensor, weight, bias, normalized_shape, eps):
    # Extract normalized_shape size for the kernel
    return (input_tensor, weight, bias, int(normalized_shape[0]) if hasattr(normalized_shape, '__getitem__') else int(normalized_shape), eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load input data
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    
    # Calculate mean
    block_mean = tl.sum(x, axis=0) / hidden_size
    block_mean = tl.sum(block_mean, axis=0) / tl.max(tl.sum(mask, axis=0), 1)
    
    # Calculate variance
    x_centered = x - block_mean
    block_var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    block_var = tl.sum(block_var, axis=0) / tl.max(tl.sum(mask, axis=0), 1)
    
    # Apply layer normalization
    normalized = x_centered / tl.sqrt(block_var + eps)
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + idx % hidden_size, mask=idx % hidden_size < hidden_size, other=1.0)
    bias = tl.load(bias_ptr + idx % hidden_size, mask=idx % hidden_size < hidden_size, other=0.0)
    
    output = normalized * weight + bias
    tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def optimized_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data in blocks for better memory access
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean across the hidden dimension for each position
    # For efficiency, we'll compute mean per block and then combine
    block_mean = tl.sum(x, axis=0) / hidden_size
    block_mean = tl.sum(block_mean, axis=0) / tl.max(tl.sum(mask, axis=0), 1)
    
    # Calculate variance
    x_centered = x - block_mean
    block_var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    block_var = tl.sum(block_var, axis=0) / tl.max(tl.sum(mask, axis=0), 1)
    
    # Apply layer normalization
    normalized = x_centered / tl.sqrt(block_var + eps)
    
    # Load weight and bias (these are smaller, so we broadcast)
    weight = tl.load(weight_ptr + offsets % hidden_size, mask=offsets % hidden_size < hidden_size, other=1.0)
    bias = tl.load(bias_ptr + offsets % hidden_size, mask=offsets % hidden_size < hidden_size, other=0.0)
    
    # Apply weight and bias
    output = normalized * weight + bias
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(input_tensor, weight, bias, hidden_size, eps):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    # Use the optimized kernel with better memory access patterns
    optimized_layer_norm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    # Return a closure that captures the arguments needed
    def layer_norm_wrapper(input_tensor, weight, bias, normalized_shape, eps):
        hidden_size = int(normalized_shape[0]) if hasattr(normalized_shape, '__getitem__') else int(normalized_shape)
        return optimized_layer_norm(input_tensor, weight, bias, hidden_size, eps)
    return layer_norm_wrapper