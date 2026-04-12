import torch
import triton
import triton.language as tl
import math

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Program id = row index
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Compute base offset for this row
    offset_base = row_idx * hidden_size
    
    # Compute mean using vectorized loads
    row_sum = 0.0
    for n in range(0, hidden_size, BLOCK_SIZE):
        # Load a block of data
        offsets = offset_base + n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < offset_base + hidden_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(x)
    
    row_mean = row_sum / hidden_size
    
    # Compute variance using vectorized loads  
    row_var_sum = 0.0
    for n in range(0, hidden_size, BLOCK_SIZE):
        # Load a block of data
        offsets = offset_base + n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < offset_base + hidden_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - row_mean
        row_var_sum += tl.sum(diff * diff)
    
    row_var = row_var_sum / hidden_size
    row_rstd = 1.0 / tl.sqrt(row_var + eps)
    
    # Normalize and apply scale/bias using vectorized stores
    for n in range(0, hidden_size, BLOCK_SIZE):
        # Load a block of data
        offsets = offset_base + n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < offset_base + hidden_size
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Apply normalization
        y = (x - row_mean) * row_rstd
        
        # Apply scale and bias if provided
        if weight_ptr is not None and bias_ptr is not None:
            # For weight and bias, we do scalar loads since they're small 1D tensors
            w_start = n
            for i in range(min(BLOCK_SIZE, hidden_size - n)):
                w = tl.load(weight_ptr + w_start + i).to(tl.float32)
                b = tl.load(bias_ptr + w_start + i).to(tl.float32)
                # Apply scale/bias element-wise
                y = tl.where(offsets + i < offset_base + hidden_size, y * w + b, y)
        else:
            # Scale and bias not provided, just normalize
            pass
        
        # Store result
        tl.store(out_ptr + offsets, y.to(tl.float16 if x.dtype == tl.float16 else tl.float32), mask=mask)



@torch.fx.wrap  
def triton_layer_norm(x, normalized_shape, weight, bias, eps):
    # Get tensor dimensions
    batch_size = x.shape[0]
    seq_len = x.shape[1] 
    hidden_size = x.shape[2]
    n_rows = batch_size * seq_len  # Each row needs independent normalization
    
    # Configure block size - should divide evenly into hidden_size (1024)
    BLOCK_SIZE = 256  # 1024 / 4 = good balance between vectorization and parallelism
    
    # Number of programs = number of rows
    num_programs = n_rows
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_rows=n_rows,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_layer_norm