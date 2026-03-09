import torch
import triton
import triton.language as tl

def pattern(tmp_6, weight, bias, eps):
    # Pattern for layer normalization: layer_norm(x, weight, bias, eps)
    return torch.nn.functional.layer_norm(tmp_6, tmp_6.shape[-1:], weight, bias, eps)

def replacement_args(tmp_6, weight, bias, eps):
    return (tmp_6, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    eps,
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    block_start = row * hidden_size + col * BLOCK_SIZE
    
    mask = block_start < n_elements & (block_start // hidden_size == row)
    
    if mask:
        # Load x data for this row
        x_offset = row * hidden_size + block_start
        x_data = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Compute mean
        sum_x = tl.sum(x_data, axis=0, mask=mask)
        mean = sum_x / hidden_size
        
        # Compute variance
        x_centered = x_data - mean
        x_centered_sq = x_centered * x_centered
        sum_x_sq = tl.sum(x_centered_sq, axis=0, mask=mask)
        var = sum_x_sq / hidden_size
        
        # Apply normalization
        denom = tl.sqrt(var + eps)
        x_norm = x_centered / denom
        
        # Load weight and bias
        weight = tl.load(weight_ptr + col, mask=col < hidden_size, other=1.0)
        bias = tl.load(bias_ptr + col, mask=col < hidden_size, other=0.0)
        
        # Apply weight and bias
        output = x_norm * weight + bias
        
        # Store result
        tl.store(output_ptr + x_offset, output, mask=mask)

@triton.jit 
def layer_norm_kernel_vec(
    x_ptr,
    weight_ptr,
    bias_ptr, 
    output_ptr,
    eps,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # More efficient 2D kernel with vectorization
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    mask = (batch_idx < batch_size) & (seq_idx < seq_len)
    
    if mask:
        # Load all data for this batch sequence position
        x_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size
        x_data = tl.load(x_ptr + x_offset, mask=mask).to(tl.float32)
        
        # Compute statistics for the entire hidden dimension
        mean = tl.sum(x_data) / hidden_size
        var = tl.sum((x_data - mean) * (x_data - mean)) / hidden_size
        
        # Normalize
        denom = tl.sqrt(var + eps)
        x_norm = (x_data - mean) / denom
        
        # Load weight and bias (broadcast)
        weight = tl.load(weight_ptr, mask=mask).to(tl.float32)
        bias = tl.load(bias_ptr, mask=mask).to(tl.float32)
        
        # Apply affine transformation
        output = x_norm * weight + bias
        
        # Store result
        tl.store(output_ptr + x_offset, output.to(tl.float32), mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-12):
    if x.numel() == 0:
        return torch.empty_like(x)
    
    batch_size = x.shape[0]
    seq_len = x.shape[1] 
    hidden_size = x.shape[-1]
    
    output = torch.empty_like(x)
    
    # Choose kernel based on hidden size
    if hidden_size <= 256:
        # Use simpler kernel for small hidden sizes
        BLOCK_SIZE = 256
        grid_size = (batch_size * seq_len, 1)
        
        layer_norm_kernel_vec[grid_size](
            x, weight, bias, output, eps,
            batch_size, seq_len, hidden_size,
            BLOCK_SIZE,
        )
    else:
        # Use vectorized kernel for larger hidden sizes
        BLOCK_SIZE = 1024
        grid_size = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        layer_norm_kernel_vec[grid_size](
            x, weight, bias, output, eps,
            batch_size, seq_len, hidden_size,
            1, 1024  # BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    
    return output

# Fallback for different eps values
def optimized_layer_norm_eps(x, weight, bias, eps):
    return optimized_layer_norm(x, weight, bias, eps)

def replacement_func():
    return optimized_layer_norm_eps