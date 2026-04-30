import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """
    Pattern to match: layer_norm followed by dropout
    dropout with training=False is essentially identity - can fuse them
    """
    ln_out = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    dropout_out = torch.nn.functional.dropout(ln_out, p=0.1, training=False)
    return ln_out, dropout_out

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

# Autotune configurations for different hidden sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
    ],
    key=['hidden_size'],
)
@triton.jit
def layer_norm_dropout_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    hidden_size, eps: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    # Get program IDs for batch and sequence dimensions
    batch_pid = tl.program_id(0)
    seq_pid = tl.program_id(1)
    
    # Calculate row offset
    row_offset = (batch_pid * tl.num_programs(1) + seq_pid) * hidden_size
    
    # Load the hidden dimension values
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_offset + hidden_size
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean across hidden dimension
    mean = tl.sum(x, axis=0) / hidden_size
    mean = tl.reshape(mean, [1])
    
    # Compute variance
    var = tl.sum((x - mean) * (x - mean), axis=0) / hidden_size
    var = tl.reshape(var, [1])
    
    # Compute std
    std = tl.sqrt(var + eps)
    
    # Normalize
    x_norm = (x - mean) / std
    
    # Load weight and bias (they're 1D)
    w_offsets = tl.arange(0, BLOCK_SIZE)
    w_mask = w_offsets < hidden_size
    w = tl.load(weight_ptr + w_offsets, mask=w_mask, other=1.0)
    b = tl.load(bias_ptr + w_offsets, mask=w_mask, other=0.0)
    
    # Apply affine transformation: output = x_norm * weight + bias
    output = x_norm * w + b
    
    # Store result (dropout with training=False is identity)
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def fused_layer_norm_dropout(x, normalized_shape, weight, bias, eps):
    """
    Fused layer_norm + dropout kernel.
    Dropout with training=False is essentially identity, so we just apply layernorm.
    """
    # Get shape info
    if isinstance(normalized_shape, tuple):
        hidden_size = normalized_shape[0]
    else:
        hidden_size = normalized_shape
    
    # Get batch and sequence dimensions
    n_elements = x.numel()
    batch_seq_size = n_elements // hidden_size
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Grid: (batch_size, seq_len)
    # This assumes x has shape [batch, seq, hidden]
    # We need to infer batch and seq from the total size
    batch_size = batch_seq_size // 11  # heuristic based on typical seq lengths
    seq_len = 11
    
    # If the above doesn't work, try a simpler grid
    if batch_size * seq_len != batch_seq_size:
        batch_size = 1
        seq_len = batch_seq_size
    
    grid = (batch_size, seq_len)
    
    layer_norm_dropout_kernel[grid](
        x, weight, bias, output,
        hidden_size, eps
    )
    
    return output, output  # Return both ln_out and dropout_out

def replacement_func():
    return fused_layer_norm_dropout