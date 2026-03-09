import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, eps):
    """Pattern to match layer normalization with normalized_shape=(768,)"""
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, eps)

def replacement_args(x, weight, bias, eps):
    """Return arguments for layer norm optimization"""
    return (x, weight, bias, eps)

@triton.jit
def layer_norm_kernel_768d(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for 768-dimensional layer normalization"""
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input block
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean for this block
    mean = tl.sum(x) / tl.sum(tl.sum(mask, axis=0))
    
    # Compute variance for this block
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / tl.sum(tl.sum(mask, axis=0))
    
    # Layer norm
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = x_centered * inv_std
    
    # Load weight and bias (broadcast across elements)
    if pid == 0:  # Only load once per block from main thread
        weight = tl.load(weight_ptr)
        bias = tl.load(bias_ptr)
    
    # Apply weight and bias
    out = out * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=2),
        triton.Config(num_warps=8, num_stages=2),
        triton.Config(num_warps=4, num_stages=3),
        triton.Config(num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def layer_norm_kernel_768d_autotune(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned Triton kernel for 768-dimensional layer normalization"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / tl.sum(tl.sum(mask, axis=0))
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / tl.sum(tl.sum(mask, axis=0))
    
    # Layer norm
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = x_centered * inv_std
    
    # Apply weight and bias (scalar broadcast)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    out = out * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm_768d(x, weight, bias, eps):
    """Optimized layer norm wrapper for 768-dimensional input"""
    # Get input shape (expected: [batch_size, seq_len, 768])
    if x.dim() != 3:
        # Fallback to original implementation for unsupported shapes
        return torch.nn.functional.layer_norm(x, (768,), weight, bias, eps)
    
    batch_size, seq_len, _ = x.shape
    n_elements = batch_size * seq_len * 768
    
    out = torch.empty_like(x)
    
    # Launch kernel with autotuning
    BLOCK_SIZE = 1024  # Optimal block size for 768-dimensional data
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    try:
        # Try autotuned version first
        layer_norm_kernel_768d_autotune[(num_programs,)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            n_elements=n_elements,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    except:
        # Fallback to simpler version
        layer_norm_kernel_768d[(num_programs,)](
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
    return optimized_layer_norm_768d