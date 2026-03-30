import torch
import triton
import triton.language as tl
import math

def pattern(x, weight, bias, eps=1e-12):
    """Pattern matching: layer normalization"""
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, eps)

def replacement_args(x, weight, bias, eps=1e-12):
    """Extract arguments needed for layer norm"""
    return (x, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,           # Input tensor pointer
    weight_ptr,      # Weight tensor pointer  
    bias_ptr,        # Bias tensor pointer
    out_ptr,         # Output tensor pointer
    mean_ptr,        # Pointer to store computed mean
    var_ptr,         # Pointer to store computed variance
    n_elements,      # Total number of elements (768 for [1, 768] tensor)
    eps: tl.constexpr,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance layer normalization kernel with proper mean/var computation"""
    pid = tl.program_id(0)
    
    if pid == 0:
        # For the first program, we need to compute global mean and variance
        # Load tensor to a small buffer for reduction
        x_vals = tl.load(x_ptr + tl.arange(0, n_elements)).to(tl.float32)
        
        # Compute mean using atomic operations or direct computation
        mean_val = tl.sum(x_vals) / n_elements
        tl.store(mean_ptr, mean_val)
        
        # Compute variance
        x_centered = x_vals - mean_val
        var_val = tl.sum(x_centered * x_centered) / n_elements
        tl.store(var_ptr, var_val)
        
        # Sync all threads to ensure mean and variance are computed
        tl.debug_barrier()  # Force synchronization
    
    # For all threads, compute normalization
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input, weight, and bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)
    
    # Apply normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    
    # Apply weight and bias
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-12):
    """Optimized layer normalization with Triton kernel"""
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Smaller block size for better utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor with same dtype and device as input
    out = torch.empty_like(x)
    
    # Create temporary buffers for mean and variance
    mean_storage = torch.empty(1, dtype=torch.float32, device=x.device)
    var_storage = torch.empty(1, dtype=torch.float32, device=x.device)
    
    # Convert x to float32 for precise computation (as in line 32 of kernel)
    x_float32 = x.to(torch.float32)
    
    # Launch kernel
    layer_norm_kernel[(num_programs,)](
        x_ptr=x_float32,
        weight_ptr=weight.to(torch.float32),  # Convert weight to float32
        bias_ptr=bias.to(torch.float32),      # Convert bias to float32
        out_ptr=out,
        mean_ptr=mean_storage,
        var_ptr=var_storage,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized layer normalization function"""
    return optimized_layer_norm