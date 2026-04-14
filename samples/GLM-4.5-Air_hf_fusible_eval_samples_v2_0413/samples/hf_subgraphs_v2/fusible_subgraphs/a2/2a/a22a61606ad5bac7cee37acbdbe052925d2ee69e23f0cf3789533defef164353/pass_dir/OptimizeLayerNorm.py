import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    """Pattern to match layer normalization operation"""
    # Normalize over the last dimension
    normalized_shape = (input_tensor.shape[-1],)
    return torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, 1e-12)

def replacement_args(input_tensor, weight, bias):
    hidden_size = input_tensor.shape[-1]
    return (input_tensor, weight, bias, hidden_size)

def replacement_func():
    return optimized_layer_norm

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    batch_size, seq_len, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized layer normalization kernel"""
    pid = tl.program_id(0)
    if pid >= batch_size * seq_len:
        return
    
    # Compute base pointers for this element
    x_base = x_ptr + pid * hidden_size
    y_base = y_ptr + pid * hidden_size
    
    # Compute mean and variance using Welford's algorithm online
    mean = 0.0
    m2 = 0.0
    
    # First pass: compute mean
    count = 0
    for h in range(0, hidden_size, BLOCK_SIZE):
        mask = h + tl.arange(0, BLOCK_SIZE) < hidden_size
        vals = tl.load(x_base + h, mask=mask, other=0.0)
        mean += tl.sum(vals)
        count += tl.sum(mask.astype(tl.float32))
    
    # Reduce across threads (simplified - in real implementation would use proper reduction)
    if count > 0:
        mean /= count
    
    # Second pass: compute variance
    for h in range(0, hidden_size, BLOCK_SIZE):
        mask = h + tl.arange(0, BLOCK_SIZE) < hidden_size
        vals = tl.load(x_base + h, mask=mask, other=0.0)
        diff = vals - mean
        m2 += tl.sum(diff * diff)
    
    # Variance
    if count > 0:
        var = m2 / count + eps
    else:
        var = eps
    
    # Compute inverse square root of variance
    inv_std = tl.sqrt(var).reciprocal()
    
    # Apply normalization, scaling, and bias
    for h in range(0, hidden_size, BLOCK_SIZE):
        mask = h + tl.arange(0, BLOCK_SIZE) < hidden_size
        
        # Load input
        x_val = tl.load(x_base + h, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x_val - mean) * inv_std
        
        # Apply scale and bias
        weight_val = tl.load(weight_ptr + h, mask=mask, other=1.0)
        bias_val = tl.load(bias_ptr + h, mask=mask, other=0.0)
        y_val = x_norm * weight_val + bias_val
        
        # Store result
        tl.store(y_base + h, y_val, mask=mask)

@triton.jit
def fast_layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    batch_size, seq_len, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Faster layer normalization kernel using parallel reduction for mean/var"""
    pid = tl.program_id(0)
    if pid >= batch_size * seq_len:
        return
    
    # Compute base pointers for this element
    x_base = x_ptr + pid * hidden_size
    y_base = y_ptr + pid * hidden_size
    
    # Load all data for this position
    x_vals = tl.load(x_base + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    
    # Mask for valid elements
    mask = tl.arange(0, BLOCK_SIZE) < hidden_size
    
    # Compute mean using reduction
    sum_x = tl.sum(x_vals * mask)
    count = tl.sum(mask)
    mean = sum_x / count if count > 0 else 0.0
    
    # Compute variance
    x_centered = x_vals - mean
    sum_x2 = tl.sum(x_centered * x_centered * mask)
    var = sum_x2 / count + eps
    
    # normalization
    inv_std = 1.0 / tl.sqrt(var)
    
    # Load weight and bias vectors (ensure they fit in BLOCK_SIZE)
    weight_vals = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    bias_vals = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    
    # Apply normalization, scaling, and bias
    x_norm = x_centered * inv_std
    y_vals = x_norm * weight_vals + bias_vals
    
    # Store result
    tl.store(y_base + tl.arange(0, BLOCK_SIZE), y_vals.to(mask.dtype), mask=mask)

@torch.fx.wrap
def optimized_layer_norm(input_tensor, weight, bias, hidden_size):
    batch_size, seq_len = input_tensor.shape[0], input_tensor.shape[1]
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Choose appropriate kernel based on hidden size
    if hidden_size <= 1024:
        # Use faster kernel for smaller hidden sizes
        BLOCK_SIZE = min(1024, hidden_size)
        grid = (batch_size * seq_len,)
        fast_layer_norm_kernel[grid](
            input_tensor, weight, bias, output,
            batch_size, seq_len, hidden_size,
            1e-12,
            BLOCK_SIZE
        )
    else:
        # Use more robust kernel for larger hidden sizes
        BLOCK_SIZE = 256
        grid = (batch_size * seq_len,)
        layer_norm_kernel[grid](
            input_tensor, weight, bias, output,
            batch_size, seq_len, hidden_size,
            1e-12,
            BLOCK_SIZE
        )
    
    return output