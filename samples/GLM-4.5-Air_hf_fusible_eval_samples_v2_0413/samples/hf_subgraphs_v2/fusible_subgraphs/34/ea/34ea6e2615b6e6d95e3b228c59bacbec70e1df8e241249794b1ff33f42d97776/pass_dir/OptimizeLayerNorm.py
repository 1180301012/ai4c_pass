import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    rows,
    cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one column across all rows
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    if col_idx >= cols or row_idx >= rows:
        return
        
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + col_idx)
    beta = tl.load(beta_ptr + col_idx)
    
    # Calculate block size for parallel reduction
    n_elements = rows
    block_start = row_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements for this column
    x = tl.load(x_ptr + offsets * cols + col_idx, mask=mask, other=0.0)
    
    # Calculate mean using parallel reduction
    sum_x = tl.sum(x, 0)
    mean_val = sum_x / rows
    
    # Calculate variance
    x_centered = x - mean_val
    x_centered_sq = x_centered * x_centered
    sum_x2 = tl.sum(x_centered_sq, 0)
    var_val = sum_x2 / rows + eps
    
    # Standard deviation
    std_val = tl.sqrt(var_val)
    
    # Normalize and scale
    x_normalized = x_centered / std_val
    y = x_normalized * gamma + beta
    
    # Store result
    tl.store(output_ptr + offsets * cols + col_idx, y, mask=mask)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias, eps=1e-05):
    # Get tensor shapes
    rows, cols = x.shape[-2], x.shape[-1]
    
    # Create output
    output = torch.empty_like(x)
    
    # Handle different input dimensions
    if len(x.shape) == 2:
        # 2D case: [rows, cols]
        rows_dim = rows
        cols_dim = cols
        grid = (cols, rows)
    elif len(x.shape) == 3:
        # 3D case: [batch, rows, cols] - apply to last two dimensions
        rows_dim = x.shape[1] * x.shape[0]  # batch * rows
        cols_dim = cols
        # Reshape to 2D for processing
        x_reshaped = x.reshape(-1, cols)
        output_reshaped = output.reshape(-1, cols)
        grid = (cols, rows_dim)
        layernorm_kernel[grid](
            x_reshaped,
            weight,
            bias,
            output_reshaped,
            rows_dim, cols_dim,
            eps,
            BLOCK_SIZE=1024,
        )
        return output
    else:
        raise ValueError("Unsupported input dimensions for layer norm")
    
    # Launch kernel
    layernorm_kernel[grid](
        x,
        weight,
        bias,
        output,
        rows_dim, cols_dim,
        eps,
        BLOCK_SIZE=1024,
    )
    
    return output

# Pattern matching function - matches layer norm
def pattern(input_tensor, weight_tensor, bias_tensor):
    tmp_8 = torch.nn.functional.layer_norm(input_tensor, (768,), weight_tensor, bias_tensor, 1e-05)
    return tmp_8

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    def optimized_forward(input_tensor, weight_tensor, bias_tensor):
        # Use optimized layer norm
        tmp_8 = optimized_layernorm(input_tensor, weight_tensor, bias_tensor, 1e-05)
        return tmp_8
    
    return optimized_forward