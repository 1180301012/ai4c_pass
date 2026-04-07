import torch
import triton
import triton.language as tl
import math

@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    n_elements,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance layer normalization kernel.
    
    Args:
        x_ptr: Input tensor pointer (features: [n_elements, hidden_dim])
        gamma_ptr: Weight tensor pointer (shape: [hidden_dim])
        beta_ptr: Bias tensor pointer (shape: [hidden_dim])
        output_ptr: Output tensor pointer (shape: [n_elements, hidden_dim])
        n_elements: Number of elements (sequence_length * batch_size)
        eps: Epsilon value for numerical stability
        BLOCK_SIZE: Block size for kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load entire feature vector for this element
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load gamma and beta vectors
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / n_elements
    
    # Calculate variance (need to adjust for actual number of features)
    sum_x2 = tl.sum(x * x, axis=0)
    variance = sum_x2 / n_elements - mean * mean
    
    # Normalize and apply scale/shift
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    y = x_norm * gamma + beta
    
    # Store output
    tl.store(output_ptr + offsets, y, mask=mask)

@triton.jit
def layer_norm_kernel_mixed(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    hidden_dim,
    n_elements,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer norm kernel for mixed dimensions (32x32 vs 64x64 spatial).
    """
    # Each program handles one element in the sequence
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and normalize per feature dimension
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean and variance along feature dimension
    x_mean = tl.sum(x) / hidden_dim
    x_var = tl.sum((x - x_mean) * (x - x_mean)) / hidden_dim
    
    # Normalize and apply scale/shift
    x_norm = (x - x_mean) / tl.sqrt(x_var + eps)
    y = x_norm * gamma + beta
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

@triton.jit
def layer_norm_autotuned_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    n_elements,
    hidden_dim,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned layer norm kernel that handles different hidden dimensions."""
    pid = tl.program_id(0)
    
    # Process each element in the sequence with autotuning consideration
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input, gamma, and beta with mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean and variance more efficiently
    x_mean = tl.sum(x * mask) / hidden_dim
    x_var = tl.sum(((x - x_mean) * (x - x_mean)) * mask) / hidden_dim
    
    # Apply normalization
    x_norm = (x - x_mean) * mask / tl.sqrt(x_var + eps)
    y = x_norm * gamma + beta
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(hidden_dim, n_elements, x, weight, bias, eps=1e-05):
    """
    Optimized layer normalization using Triton.
    
    Args:
        hidden_dim: Feature dimension size
        n_elements: Number of elements in sequence
        x: Input tensor [n_elements, hidden_dim]
        weight: Weight tensor [hidden_dim]
        bias: Bias tensor [hidden_dim]
        eps: Epsilon value
    """
    # Choose optimal block size based on hidden dimension
    if hidden_dim == 768:
        BLOCK_SIZE = 1024
    elif hidden_dim == 384:
        BLOCK_SIZE = 768
    else:
        BLOCK_SIZE = 512
    
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch optimized kernel
    layer_norm_autotuned_kernel[grid_size](
        x,
        weight,
        bias,
        output,
        n_elements,
        hidden_dim,
        eps,
        BLOCK_SIZE
    )
    
    return output

def pattern(tmp_5, in_1, in_0):
    # Pattern matches layer norm with hardcoded dimension (to match exact model)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    return tmp_5, tmp_6

def replacement_args(tmp_5, in_1, in_0):
    return (tmp_5, in_1, in_0,)

def replacement_func():
    def layer_norm_wrapper(tmp_5, in_1, in_0):
        # Get dimensions
        hidden_dim = tmp_5.shape[-1]  # 768 or 384
        n_elements = tmp_5.numel() // hidden_dim
        
        # Apply optimized layer norm
        output = optimized_layer_norm(hidden_dim, n_elements, tmp_5, in_1, in_0, 1e-05)
        return output
    
    return layer_norm_wrapper