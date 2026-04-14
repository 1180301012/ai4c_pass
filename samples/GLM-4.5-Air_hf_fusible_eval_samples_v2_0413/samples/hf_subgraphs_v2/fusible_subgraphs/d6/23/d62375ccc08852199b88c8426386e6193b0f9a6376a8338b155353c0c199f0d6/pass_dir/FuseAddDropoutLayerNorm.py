import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3):
    """Fuse addition, dropout, and layer normalization operations"""
    tmp_12 = in_0 + in_1
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, in_1.shape[1:], in_3, in_2, 1e-05)
    return tmp_13, tmp_14

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, in_1.shape[1])

@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    prob: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply dropout using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random numbers using XorShift32
    rand_state = tl.arange(0, BLOCK_SIZE) + pid * 1024
    for i in range(4):  # 4 rounds of XorShift
        rand_state = rand_state ^ (rand_state << 3)
        rand_state = rand_state ^ (rand_state >> 11)
        rand_state = rand_state ^ (rand_state << 15)
    
    # Convert to [0, 1] range
    rand = rand_state / (2**32 - 1)
    mask_keep = rand > prob
    
    # Apply dropout: scale up remaining values to maintain expected sum
    scale_factor = 1.0 / (1.0 - prob)
    out = x * mask_keep * scale_factor
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def rms_norm(x, weight, eps):
    """RMS Normalization implementation"""
    rstd = 1.0 / torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return x * rstd * weight

# Triton kernel for layer norm
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Layer normalization kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets % hidden_size, mask=mask)
    if bias_ptr != 0:
        bias = tl.load(bias_ptr + offsets % hidden_size, mask=mask)
    else:
        bias = 0.0
    
    # Calculate mean and variance in blocks (simplified)
    # For production, this would need a more sophisticated two-pass approach
    # Here we use a simplified version for demonstration
    
    # First pass: calculate sum of squares
    x_sum = tl.sum(x if mask else 0.0)
    x_sq_sum = tl.sum((x * x) if mask else 0.0)
    
    # Convert to float for proper scaling
    n = tl.sum(mask, dtype=tl.float32)
    mean = x_sum / n if n > 0 else 0.0
    var = (x_sq_sum / n - mean * mean) if n > 0 else 1.0
    
    # Calculate inverse std
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization
    out = (x - mean) * rstd * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def fused_add_dropout_layer_norm(x, emb, weight, bias, hidden_size, eps=1e-05):
    """Fused implementation of addition + dropout + layer normalization"""
    # Addition
    added = x + emb
    
    # Allocation for dropout result
    n_elements = added.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    dropout_result = torch.empty_like(added)
    
    # Apply dropout
    dropout_kernel[(num_programs,)](
        added,
        dropout_result,
        n_elements,
        0.1,  # dropout probability
        BLOCK_SIZE,
    )
    
    # Apply layer normalization
    # For production, this would need a more sophisticated implementation
    # Using PyTorch's built-in layer_norm for now as Triton LN is complex
    ln_result = torch.nn.functional.layer_norm(dropout_result, (hidden_size,), bias, weight, eps)
    
    return dropout_result, ln_result

def replacement_func():
    return fused_add_dropout_layer_norm