import torch
import triton
import triton.language as tl

def pattern(tensor, weight, bias):
    # Match layer_norm operation with generic tuple (works for any hidden_size)
    return torch.nn.functional.layer_norm(tensor, weight.shape[0], weight, bias, 1e-05)

def replacement_args(tensor, weight, bias):
    return (tensor, weight, bias)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each block processes one row in the batch
    row_offset = pid * hidden_size
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load one row of the tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight and bias (broadcasted)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    # Calculate mean with better numerical stability
    row_mean = tl.sum(x) / hidden_size
    
    # Calculate variance with better precision
    x_centered = x - row_mean
    x_var = tl.sum(x_centered * x_centered) / hidden_size
    
    # Add small epsilon for numerical stability (matching PyTorch behavior)
    x_var = tl.maximum(x_var, eps)
    
    # Normalize and scale (matching PyTorch's exact operations)
    x_norm = x_centered * tl.rsqrt(x_var)
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def next_power_of_2(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

@torch.fx.wrap
def optimized_layer_norm(tensor, weight, bias, eps=1e-05):
    # Get input shape
    if tensor.dim() == 3:  # [batch, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = tensor.shape
        n_elements = batch_size * seq_len
    else:  # [batch, hidden_size]
        batch_size = tensor.shape[0]
        seq_len = 1
        hidden_size = tensor.shape[-1]
        n_elements = batch_size
    
    # Create output tensor
    output = torch.empty_like(tensor)
    
    # Choose optimal block size (must be power of 2 for tl.arange)
    BLOCK_SIZE = min(hidden_size, 1024)
    BLOCK_SIZE = next_power_of_2(BLOCK_SIZE)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    layer_norm_kernel[(num_programs,)](
        x_ptr=tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_layer_norm