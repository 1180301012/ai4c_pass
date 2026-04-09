import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, weight, bias, eps):
    """
    Match the layer normalization pattern:
    torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    """
    result = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return result

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Optimized layer normalization kernel
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    n_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized layer normalization kernel with fused affine transformation
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input, weight, and bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets % n_features, mask=offsets % n_features < n_features).to(tl.float32)
    bias = tl.load(bias_ptr + offsets % n_features, mask=offsets % n_features < n_features).to(tl.float32)
    
    # Reshape for per-feature operations
    x = x.view(-1, n_features)
    weight = weight.view(-1)
    bias = bias.view(-1)
    
    # Compute mean
    rows = x.shape[0]
    mean = tl.sum(x, axis=1) / rows
    
    # Compute variance
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) / rows
    
    # Normalize and apply affine transformation
    x_norm = (x_centered) / tl.sqrt(var[:, None] + eps)
    output = x_norm * weight[None, :] + bias[None, :]
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-05):
    """
    Optimized layer normalization with Triton kernel
    Fuses normalization and affine transformation operations
    """
    device = x.device
    n_elements = x.numel()
    n_features = x.shape[-1]
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x, dtype=torch.float32)
    
    # Handle data type conversion - original might be in bfloat16, but layer norm typically uses float32
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    if bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    
    layer_norm_kernel[(num_programs,)](
        x,
        weight, 
        bias,
        output,
        n_elements,
        n_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    # Return a function that takes x, weight, bias and calls optimized_layer_norm
    def wrapper(x, weight, bias, eps=1e-05):
        return optimized_layer_norm(x, weight, bias, eps)
    return wrapper