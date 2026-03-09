import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, eps):
    """Pattern to match layer normalization with normalized_shape=(32,)"""
    return torch.nn.functional.layer_norm(x, (32,), weight, bias, eps)

def replacement_args(x, weight, bias, eps):
    """Return arguments for layer norm optimization"""
    return (x, weight, bias, eps)

@triton.jit
def layer_norm_kernel_32d(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for 32-dimensional layer normalization"""
    # Each program handles one element (sequence position)
    pid = tl.program_id(0)
    offset = pid * 32
    
    # Ensure we don't go out of bounds
    if offset + 32 > n_elements:
        return
    
    # Load input (32 elements per thread)
    x = tl.load(x_ptr + offset, mask=tl.arange(0, 32) < (n_elements - offset), other=0.0)
    
    # Load weight and bias (broadcast to 32 elements)
    weight = tl.load(weight_ptr + tl.arange(0, 32), mask=tl.arange(0, 32) < 32, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, 32), mask=tl.arange(0, 32) < 32, other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / 32.0
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / 32.0
    
    # Layer norm
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x_centered * inv_std) * weight + bias
    
    # Store result
    tl.store(out_ptr + offset, out, mask=tl.arange(0, 32) < (n_elements - offset))

@torch.fx.wrap
def optimized_layer_norm_32d(x, weight, bias, eps):
    """Optimized layer norm wrapper for 32-dimensional input"""
    # Get input shape (expected: [batch_size, seq_len, 32])
    if x.dim() != 3:
        # Fallback to original implementation for unsupported shapes
        return torch.nn.functional.layer_norm(x, (32,), weight, bias, eps)
    
    batch_size, seq_len, _ = x.shape
    n_elements = batch_size * seq_len * 32
    
    out = torch.empty_like(x)
    
    # Launch kernel
    BLOCK_SIZE = 32
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    layer_norm_kernel_32d[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm_32d