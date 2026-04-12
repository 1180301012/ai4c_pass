import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_2, normalized_shape, weight, bias, eps):
    result = torch.nn.functional.layer_norm(tmp_2, normalized_shape, weight, bias, eps)
    return result

# Argument extraction function
def replacement_args(tmp_2, normalized_shape, weight, bias, eps):
    return (tmp_2, normalized_shape, weight, bias, eps)

# Optimized LayerNorm kernel
@triton.jit
def layernorm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, n_elements, d_model, eps,
                    BLOCK_SIZE: tl.constexpr):
    # Each program handles one entire row (sequence position) of the batch
    row_idx = tl.program_id(0)
    
    # Calculate pointer offsets for this row
    x_ptr_row = x_ptr + row_idx * d_model
    out_ptr_row = out_ptr + row_idx * d_model
    
    # Load weight and bias (these are the same for all rows)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    # For small rows, we can handle them entirely in one program
    # if d_model <= BLOCK_SIZE:
    #     # Load entire row into registers
    #     x = tl.load(x_ptr_row).to(tl.float32)
    #     
    #     # Compute mean and variance
    #     mean = tl.sum(x) / d_model
    #     x2 = x * x
    #     var = tl.sum(x2) / d_model - mean * mean
    #     std = tl.sqrt(var + eps)
    #     
    #     # Apply layer norm
    #     x_norm = (x - mean) / std
    #     y = x_norm * weight + bias
    #     
    #     # Store result
    #     tl.store(out_ptr_row, y)
    # else:
    #     # Fall back to original approach for larger rows
    #     sum_x = 0.0
    #     sum_x2 = 0.0
    #     for i in range(0, d_model, BLOCK_SIZE):
    #         off = i + tl.arange(0, BLOCK_SIZE)
    #         mask = off < d_model
    #         x = tl.load(x_ptr_row + off, mask=mask, other=0.0)
    #         sum_x += tl.sum(x)
    #         sum_x2 += tl.sum(x * x)
    #     mean = sum_x / d_model
    #     var = (sum_x2 / d_model) - (mean * mean)
    #     std = tl.sqrt(var + eps)
    #     for i in range(0, d_model, BLOCK_SIZE):
    #         off = i + tl.arange(0, BLOCK_SIZE)
    #         mask = off < d_model
    #         x = tl.load(x_ptr_row + off, mask=mask, other=0.0)
    #         x_norm = (x - mean) / std
    #         y = x_norm * weight + bias
    #         tl.store(out_ptr_row + off, y, mask=mask)
    
    # Simplified approach: since each program handles one row entirely,
    # and we haveBLOCK_SIZE >= d_model, we can load the whole row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_model
    
    # Load entire row (with masking)
    x = tl.load(x_ptr_row + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean and variance
    sum_x = tl.sum(x)
    sum_x2 = tl.sum(x * x)
    mean = sum_x / d_model
    var = sum_x2 / d_model - mean * mean
    std = tl.sqrt(var + eps)
    
    # Apply layer normalization and scale/shift
    x_norm = (x - mean) / std
    y = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr_row + offsets, y, mask=mask)

# Optimized LayerNorm kernel
    # For optimal performance, we need to check if the entire row fits in registers
    row_idx = tl.program_id(0)
    
    # Calculate pointer offsets for this row
    x_ptr_row = x_ptr + row_idx * d_model
    out_ptr_row = out_ptr + row_idx * d_model
    
    # Load weight and bias (scalar for each dim)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    # For small d_model (like 128), we can process the entire row at once
    if d_model <= BLOCK_SIZE:
        # Load entire row into registers efficiently
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < d_model
        
        # Load and convert to float32 for precision
        x = tl.load(x_ptr_row + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Single-pass mean and variance computation
        # Use more stable formulas for numerical accuracy
        mean = tl.sum(x, axis=0) / d_model
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / d_model
        std = tl.sqrt(var + eps)
        
        # Layer norm: (x - mean) / std * weight + bias
        x_norm = x_centered / std
        y = x_norm * weight + bias
        
        # Store result in original dtype
        tl.store(out_ptr_row + offsets, y, mask=mask)
    else:
        # Fallback for larger dimensions (not needed for our case)
        pass

# Autotuned kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_layernorm_autotuned(x, normalized_shape, weight, bias, eps):
    # Create output tensor
    out = torch.empty_like(x)
    
    d_model = x.shape[-1]
    batch_size = x.shape[0] if len(x.shape) > 2 else 1
    seq_len = x.shape[1] if len(x.shape) > 2 else (x.shape[0] if len(x.shape) > 1 else 1)
    num_rows = batch_size * seq_len
    
    # Use optimal block size based on our d_model (128)
    BLOCK_SIZE = max(128, d_model * 2)  # Ensure we cover the row size
    
    # Launch Triton kernel
    grid = (num_rows,)
    METADATA = 0  # Metadata flag for future optimizations
    
    layernorm_kernel_optimized[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=x.numel(),
        d_model=d_model,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        METADATA=METADATA
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_layernorm_autotuned