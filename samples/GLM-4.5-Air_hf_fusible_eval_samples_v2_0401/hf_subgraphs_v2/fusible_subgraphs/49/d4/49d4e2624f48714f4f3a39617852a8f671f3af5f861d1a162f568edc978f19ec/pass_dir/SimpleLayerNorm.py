import torch
import triton
import triton.language as tl

def pattern(tmp_3, normalized_shape, in_1, in_0, eps):
    """Pattern matches layer_norm operation"""
    result = torch.nn.functional.layer_norm(tmp_3, normalized_shape, in_1, in_0, eps)
    return result

def replacement_args(tmp_3, normalized_shape, in_1, in_0, eps):
    """Extract arguments for the optimized kernel"""
    return (tmp_3, normalized_shape, in_1, in_0, eps)

@triton.jit
def simple_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    normalized_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple LayerNorm kernel"""
    pid = tl.program_id(0)
    offset = pid * normalized_size
    
    # Load weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, normalized_size))
    bias = tl.load(bias_ptr + tl.arange(0, normalized_size))
    
    # Load input
    x = tl.load(input_ptr + offset + tl.arange(0, normalized_size))
    
    # Compute mean
    mean = tl.sum(x) / normalized_size
    
    # Compute variance
    var = tl.sum((x - mean) * (x - mean)) / normalized_size
    
    # Normalize and apply weight/bias
    y = (x - mean) / tl.sqrt(var + eps) * weight + bias
    
    # Store output
    tl.store(output_ptr + offset + tl.arange(0, normalized_size), y)

@torch.fx.wrap
def simple_layernorm(input_tensor, normalized_shape, weight, bias, eps=1e-12):
    """Simple LayerNorm implementation"""
    normalized_size = normalized_shape[0]
    
    # Handle batch dimension - for [1, 768] input, we have 1 batch element
    if input_tensor.dim() == 2:
        batch_size = input_tensor.shape[0]
        grid = (batch_size,)
    else:
        grid = (1,)
    
    output = torch.empty_like(input_tensor)
    
    simple_layernorm_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        normalized_size=normalized_size,
        eps=eps,
        BLOCK_SIZE=1,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return simple_layernorm