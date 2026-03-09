import torch
import triton
import triton.language as tl

def pattern(input, weight, bias, normalized_shape, eps):
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_1, in_0, (128,), 1e-05)

@triton.jit
def layer_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_elements, n_features,
    eps: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean
    block_mean = tl.sum(x, axis=0) / n_features
    
    # Calculate variance
    x_centered = x - block_mean
    block_var = tl.sum(x_centered * x_centered, axis=0) / n_features
    
    # Normalize
    std = tl.sqrt(block_var + eps)
    x_normalized = x_centered / std
    
    # Apply weight and bias
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE) % n_features, mask=mask)
        x_normalized = x_normalized * weight
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE) % n_features, mask=mask)
        x_normalized = x_normalized + bias
    
    # Store output
    tl.store(output_ptr + offsets, x_normalized, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(input, weight, bias, n_features, eps=1e-05):
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input)
    
    layer_norm_kernel[(num_programs,)](
        input, weight, bias, output,
        n_elements, n_features,
        eps, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_layer_norm