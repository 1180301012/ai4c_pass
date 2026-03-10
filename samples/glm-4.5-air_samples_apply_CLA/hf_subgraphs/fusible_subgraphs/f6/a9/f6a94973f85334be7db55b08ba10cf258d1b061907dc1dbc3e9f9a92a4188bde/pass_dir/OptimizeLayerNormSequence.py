import torch
import triton
import triton.language as tl

def pattern(x, bias1, weight1, bias2, weight2):
    # First layer norm (after dropout optimization)
    tmp_15 = torch.nn.functional.layer_norm(x, (768,), bias1, weight1, 1e-05)
    # Second layer norm
    tmp_16 = torch.nn.functional.layer_norm(tmp_15, (768,), bias2, weight2, 1e-05)
    return tmp_15, tmp_16

def replacement_args(x, bias1, weight1, bias2, weight2):
    return (x, bias1, weight1, bias2, weight2)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    normalized_shape: tl.constexpr,
    eps: tl.constexpr = 1e-05,
):
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load input, bias, and weight
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    
    # Calculate mean
    mean = tl.sum(x) / n_elements
    
    # Calculate variance
    variance = tl.sum((x - mean) * (x - mean)) / n_elements
    std = tl.sqrt(variance + eps)
    
    # Normalize and apply affine transformation
    out = (x - mean) / std * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_layer_norm(x, weight, bias, eps=1e-05):
    # Get input dimensions
    shape = x.shape
    n_elements = x.numel()
    
    # Flatten input except for last dimension
    x_flat = x.reshape(-1, shape[-1])
    n_elements = x_flat.numel()
    
    # Create output
    out = torch.empty_like(x_flat)
    
    # Launch kernel
    layer_norm_kernel[(n_elements + 1023) // 1024,](
        x_ptr=x_flat,
        bias_ptr=bias,
        weight_ptr=weight,
        out_ptr=out,
        n_elements=x_flat.numel(),
        normalized_shape=shape[-1],
        eps=eps,
    )
    
    # Reshape back to original shape
    return out.reshape(shape)

@triton.jit  
def dual_layer_norm_kernel(
    x_ptr,
    bias1_ptr,
    weight1_ptr,
    bias2_ptr,
    weight2_ptr,
    out1_ptr,
    out2_ptr,
    n_elements: tl.constexpr,
    normalized_shape: tl.constexpr,
    eps: tl.constexpr = 1e-05,
):
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # First layer norm
    mean1 = tl.sum(x) / n_elements
    variance1 = tl.sum((x - mean1) * (x - mean1)) / n_elements
    std1 = tl.sqrt(variance1 + eps)
    bias1 = tl.load(bias1_ptr + offsets, mask=mask, other=0.0)
    weight1 = tl.load(weight1_ptr + offsets, mask=mask, other=1.0)
    out1 = (x - mean1) / std1 * weight1 + bias1
    
    # Second layer norm
    mean2 = tl.sum(out1) / n_elements
    variance2 = tl.sum((out1 - mean2) * (out1 - mean2)) / n_elements
    std2 = tl.sqrt(variance2 + eps)
    bias2 = tl.load(bias2_ptr + offsets, mask=mask, other=0.0)
    weight2 = tl.load(weight2_ptr + offsets, mask=mask, other=1.0)
    out2 = (out1 - mean2) / std2 * weight2 + bias2
    
    # Store results
    tl.store(out1_ptr + offsets, out1, mask=mask)
    tl.store(out2_ptr + offsets, out2, mask=mask)

@torch.fx.wrap
def fused_dual_layer_norm(x, weight1, bias1, weight2, bias2, eps=1e-05):
    # Get input dimensions
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    n_elements = x_flat.numel()
    
    # Create outputs
    out1 = torch.empty_like(x_flat)
    out2 = torch.empty_like(x_flat)
    
    # Launch kernel
    grid = (n_elements + 1023) // 1024
    dual_layer_norm_kernel[grid,](
        x_ptr=x_flat,
        bias1_ptr=bias1,
        weight1_ptr=weight1,
        bias2_ptr=bias2,
        weight2_ptr=weight2,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        normalized_shape=shape[-1],
        eps=eps,
    )
    
    # Reshape back to original shape
    return out1.reshape(shape), out2.reshape(shape)

def replacement_func():
    return fused_dual_layer_norm