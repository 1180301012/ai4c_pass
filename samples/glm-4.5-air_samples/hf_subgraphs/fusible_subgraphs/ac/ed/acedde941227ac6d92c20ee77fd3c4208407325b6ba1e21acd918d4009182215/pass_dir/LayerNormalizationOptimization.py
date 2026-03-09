import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, normalized_shape, eps):
    # Match the layer_norm pattern
    result = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    return result

def replacement_args(x, weight, bias, normalized_shape, eps):
    return (x, weight, bias, normalized_shape, eps)

@triton.jit
def ln_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    n_cols, n_rows,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each row gets processed by one program
    row_offset = pid * n_cols
    
    # Initialize variables for mean and variance calculation
    row_sum = 0.0
    row_sum_sq = 0.0
    
    # Compute sum and sum of squares for variance using vectorized operations
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (i + tl.arange(0, BLOCK_SIZE)) < n_cols
        offsets = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        
        # Compute partial sums
        partial_sum = tl.sum(x)
        partial_sum_sq = tl.sum(x * x)
        
        # Atomically add to global sums (for now, simpler approach)
        row_sum += partial_sum
        row_sum_sq += partial_sum_sq
    
    # Compute mean and variance
    mean = row_sum / n_cols
    variance = (row_sum_sq / n_cols) - (mean * mean)
    variance = tl.maximum(variance, eps)
    
    # Second pass: normalize and apply weight and bias
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (i + tl.arange(0, BLOCK_SIZE)) < n_cols
        offsets = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        
        # Normalize and apply weight and bias
        x_normalized = (x - mean) / tl.sqrt(variance)
        
        # Load weight and bias
        weight = tl.load(weight_ptr, mask=None)
        bias = tl.load(bias_ptr, mask=None)
        
        # Apply weight and bias
        out = x_normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + row_offset + offsets, out, mask=mask)

@torch.fx.wrap  
def triton_layer_norm(x, weight, bias, normalized_shape, eps=1e-05):
    # Get input tensor info
    n_cols = normalized_shape[0]  # 2560
    n_rows = x.numel() // n_cols   # 128 or 256
    
    # Use a power-of-2 BLOCK_SIZE for better GPU performance
    BLOCK_SIZE = 1024  # Power of 2: 2^10
    num_programs = n_rows
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel
    ln_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_cols=n_cols,
        n_rows=n_rows,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_layer_norm