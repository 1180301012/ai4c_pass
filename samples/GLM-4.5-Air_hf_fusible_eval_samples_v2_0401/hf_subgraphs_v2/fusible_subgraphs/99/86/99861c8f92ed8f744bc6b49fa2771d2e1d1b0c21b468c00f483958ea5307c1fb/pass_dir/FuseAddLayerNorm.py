import torch
import triton
import triton.language as tl

def pattern(weight, bias, x1, x2):
    """Match the complete computation structure"""
    # Addition operation
    added = x1 + x2
    
    # Layer normalization
    normed = torch.nn.functional.layer_norm(added, (1024,), weight, bias, 1e-05)
    
    # Return both results (order may vary across models)
    return added, normed

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_add_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    out = x1 + x2
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def fused_add_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    x1_ptr,
    x2_ptr,
    out_add_ptr,
    out_norm_ptr,
    n_elements,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # Addition
    added = x1 + x2
    tl.store(out_add_ptr + offsets, added, mask=mask)
    
    # Layer normalization
    mean = tl.sum(added, axis=0) / tl.sum(mask)
    centered = added - mean
    var = tl.sum(centered * centered, axis=0) / tl.sum(mask) + eps
    inv_std = tl.pow(var, -0.5)
    normalized = centered * inv_std
    result = normalized * weight + bias
    
    tl.store(out_norm_ptr + offsets, result, mask=mask)

@triton.jit
def simple_add_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    out = x1 + x2
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_layernorm(bias, weight, x1, x2):
    # Get input shapes
    batch_size, seq_len, hidden_dim = x1.shape
    
    # Flatten for processing
    n_elements = batch_size * seq_len * hidden_dim
    x1_flat = x1.view(-1)
    x2_flat = x2.view(-1)
    
    # Create output tensors
    out_add = torch.empty_like(x1_flat)
    out_norm = torch.empty_like(x1_flat)
    
    # Launch fused kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_layernorm_kernel[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        x1_ptr=x1_flat,
        x2_ptr=x2_flat,
        out_add_ptr=out_add,
        out_norm_ptr=out_norm,
        n_elements=n_elements,
        hidden_size=hidden_dim,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape results back to original dimensions
    out_add_reshaped = out_add.view(batch_size, seq_len, hidden_dim)
    out_norm_reshaped = out_norm.view(batch_size, seq_len, hidden_dim)
    
    return out_add_reshaped, out_norm_reshaped

def replacement_func():
    return fused_add_layernorm