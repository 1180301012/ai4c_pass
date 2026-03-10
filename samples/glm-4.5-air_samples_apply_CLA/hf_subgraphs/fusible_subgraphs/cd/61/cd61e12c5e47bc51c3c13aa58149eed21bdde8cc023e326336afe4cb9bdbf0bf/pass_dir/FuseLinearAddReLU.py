import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, residual):
    # Linear: x @ weight.T + bias (where x is [1000, 128], weight is [128, 128], bias is [128])
    tmp_1 = weight
    tmp_0 = bias
    linear_out = torch.nn.functional.linear(x, tmp_1, tmp_0)
    
    # Addition: residual + linear_out
    add_out = residual + linear_out
    
    # ReLU activation
    relu_out = add_out.relu_()
    
    return relu_out

def replacement_args(x, weight, bias, residual):
    return (x, weight, bias, residual)

@triton.jit
def linear_add_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    n_rows,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program ID determines which row we process
    row_id = tl.program_id(0)
    
    # Early exit if out of bounds
    if row_id >= n_rows:
        return
    
    # Initialize output and load bias
    result = tl.zeros((N_COLS,), dtype=tl.float32)
    bias = tl.load(bias_ptr + tl.arange(0, N_COLS))
    result += bias
    
    # Process matrix multiplication element by element
    for k in range(N_COLS):
        # Load k-th input element as scalar
        x_elem = tl.load(x_ptr + row_id * N_COLS + k)
        
        # Load k-th weight row
        weight_row = tl.load(weight_ptr + k * N_COLS + tl.arange(0, N_COLS))
        
        # Add contribution: x_elem * weight_row
        result += x_elem * weight_row
    
    # Add residual
    residual_row = tl.load(residual_ptr + row_id * N_COLS + tl.arange(0, N_COLS))
    result += residual_row
    
    # Apply ReLU
    final_result = tl.where(result > 0, result, 0)
    
    # Store result
    out_row_ptr = out_ptr + row_id * N_COLS + tl.arange(0, N_COLS)
    tl.store(out_row_ptr, final_result)

@torch.fx.wrap
def fused_linear_add_relu(x, weight, bias, residual):
    # Move weights and bias to GPU for computation
    weight_gpu = weight.to(x.device)
    bias_gpu = bias.to(x.device)
    
    n_rows, n_cols = x.shape
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Use single program per row for maximum parallelism
    # This gives us better GPU utilization despite launch overhead
    grid_rows = n_rows  # One program per row
    
    # Launch kernel with maximum parallelism (BLOCK_SIZE_M=1 for single row per program)
    linear_add_relu_kernel[(grid_rows,)](
        x_ptr=x,
        weight_ptr=weight_gpu,
        bias_ptr=bias_gpu,
        residual_ptr=residual,
        out_ptr=out,
        n_rows=n_rows,
        N_COLS=n_cols,  # Pass columns size as constant
        BLOCK_SIZE_M=1,  # Each program processes exactly one row
    )
    
    return out

def replacement_func():
    return fused_linear_add_relu