import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """Pattern matching for layer_norm operation"""
    return (x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    """Extract arguments for layer norm replacement"""
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_features: tl.constexpr,
    n_elements: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance layer norm implementation"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with proper type handling
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    
    # Load weight and bias with proper broadcasting
    weight_idx = offsets % n_features
    bias_idx = offsets % n_features
    weight = tl.load(weight_ptr + weight_idx, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean (simple approach - per-block)
    block_sum = tl.sum(x_f32)
    block_size = tl.sum(mask.to(tl.int32))
    
    if block_size > 0:
        block_mean = block_sum / block_size
    else:
        block_mean = 0.0
    
    # Compute variance
    x_centered = x_f32 - block_mean
    x_centered_sq = x_centered * x_centered
    block_var = tl.sum(x_centered_sq)
    
    if block_size > 0:
        block_var = block_var / block_size
    else:
        block_var = 1.0
    
    # Normalize and apply weight/bias
    std = tl.sqrt(block_var + eps)
    out_f32 = (x_centered / std) * weight + bias
    
    # Store results with original dtype
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, normalized_shape, eps):
    """Wrapper for optimized layer norm"""
    # Ensure tensors are on GPU for Triton
    if x.device.type == 'cpu':
        x = x.cuda()
    if weight.device.type == 'cpu':
        weight = weight.cuda()
    if bias.device.type == 'cpu':
        bias = bias.cuda()
    
    # Ensure weight and bias are properly shaped for broadcasting
    if len(weight.shape) == 1:
        # Expand weight and bias to match the last dimensions
        weight = weight.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
    if len(bias.shape) == 1:
        bias = bias.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimize for typical GPU architecture
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_features=x.shape[-1],
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Replacement function that returns the optimized layer norm"""
    return optimized_layer_norm