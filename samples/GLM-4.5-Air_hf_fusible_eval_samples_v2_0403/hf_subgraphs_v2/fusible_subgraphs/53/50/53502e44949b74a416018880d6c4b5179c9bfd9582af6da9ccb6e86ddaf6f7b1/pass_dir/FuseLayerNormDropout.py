import torch
import triton
import triton.language as tl
import math

def pattern(embeddings, gamma, beta, dropout_p=0.1):
    """Match layer_norm + dropout operation for small tensors"""
    # Layer norm
    normalized = torch.nn.functional.layer_norm(embeddings, (256,), gamma, beta, 1e-05)
    
    # Dropout
    dropped_out = torch.nn.functional.dropout(normalized, p=dropout_p, training=False)
    return dropped_out

def replacement_args(embeddings, gamma, beta, dropout_p=0.1):
    return (embeddings, gamma, beta, dropout_p)

# Triton kernel for fused layer norm + dropout
@triton.jit
def fused_layernorm_dropout_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    hidden_size,
    dropout_p: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program handles one sequence position
    pid = tl.program_id(0)
    block_size = 256
    
    # Compute base address for this position
    base_offset = pid * hidden_size
    
    # Load gamma and beta (1D parameters)
    gamma = tl.load(gamma_ptr + tl.arange(0, hidden_size))
    beta = tl.load(beta_ptr + tl.arange(0, hidden_size))
    
    # Initialize reduction variables
    sum_x = 0.0
    sum_x2 = 0.0
    
    # First pass: compute mean and variance
    for i in range(0, hidden_size, block_size):
        offsets = i + tl.arange(0, block_size)
        mask = offsets < hidden_size
        
        # Load input
        x = tl.load(input_ptr + base_offset + offsets, mask=mask)
        
        # Update reduction
        sum_x += tl.sum(x, mask=mask)
        sum_x2 += tl.sum(x * x, mask=mask)
    
    # Compute mean and variance
    mean = sum_x / hidden_size
    var = (sum_x2 / hidden_size) - (mean * mean)
    var = tl.maximum(var, eps)
    std = tl.sqrt(var)
    
    # Second pass: apply normalization and dropout
    for i in range(0, hidden_size, block_size):
        offsets = i + tl.arange(0, block_size)
        mask = offsets < hidden_size
        
        # Load input
        x = tl.load(input_ptr + base_offset + offsets, mask=mask)
        
        # Apply layer norm
        x_norm = (x - mean) / std
        y = x_norm * gamma + beta
        
        # Apply dropout (training=False, so just scaling)
        if dropout_p > 0.0:
            scale = 1.0 / (1.0 - dropout_p)
            y = y * scale
        
        # Store result
        tl.store(output_ptr + base_offset + offsets, y, mask=mask)

@torch.fx.wrap
def fused_layernorm_dropout(input, gamma, beta, dropout_p=0.1):
    batch_size, seq_len, hidden_size = input.shape
    
    # Create output tensor
    output = torch.zeros_like(input)
    
    # Launch kernel
    grid = (seq_len,)
    fused_layernorm_dropout_kernel[grid](
        input,
        gamma,
        beta,
        output,
        hidden_size,
        dropout_p,
        1e-05
    )
    
    return output

def replacement_func():
    return lambda embeddings, gamma, beta, dropout_p: fused_layernorm_dropout(embeddings, gamma, beta, dropout_p)