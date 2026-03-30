import torch
import triton
import triton.language as tl
import math

def pattern(x, in_1, in_0):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], in_1, in_0, 1e-05)

def replacement_args(x, in_1, in_0):
    return (x, in_1, in_0)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one hidden dimension
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x for this program's hidden dimension (across all batches and sequences)
    x = tl.load(x_ptr + offsets * hidden_size, mask=mask, other=0.0)
    
    # Load weight and bias for this hidden dimension
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / tl.sum(mask)
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered) / tl.sum(mask)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize and apply weight/bias
    output = (x_centered * inv_std) * weight + bias
    
    # Store result
    tl.store(output_ptr + offsets * hidden_size, output, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, in_1, in_0):
    # Handle different input shapes: (1, S, H) where S=sequence_length, H=hidden_size
    if x.dim() == 3:
        batch_size, seq_len, hidden_size = x.shape
        n_elements = seq_len
    else:
        n_elements = x.numel() // x.shape[-1]
        hidden_size = x.shape[-1]
    
    output = torch.empty_like(x)
    
    # Optimized block size for layer normalization
    BLOCK_SIZE = 1024  # Tune based on hidden_size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    layer_norm_kernel[(num_programs,)](
        x,
        in_1,
        in_0,
        output,
        n_elements,
        hidden_size,
        1e-05,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_layer_norm