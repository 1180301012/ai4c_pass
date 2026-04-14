import torch
import triton
import triton.language as tl

def pattern(in_x, in_weight, in_bias, ln_weight, ln_bias):
    # Linear transformation
    linear_out = torch.nn.functional.linear(in_x, in_weight, in_bias)
    
    # Layer norm
    ln_out = torch.nn.functional.layer_norm(linear_out, (linear_out.shape[-1],), ln_weight, ln_bias, 1e-05)
    
    # Sigmoid activation
    sigmoid_out = torch.sigmoid(ln_out)
    
    # Return all outputs that might be observable
    return ln_out, sigmoid_out

def replacement_args(in_x, in_weight, in_bias, ln_weight, ln_bias):
    return (in_x, in_weight, in_bias, ln_weight, ln_bias)

@triton.jit
def linear_layer_norm_sigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr, 
    ln_weight_ptr, ln_bias_ptr,
    out_ptr, sigmoid_out_ptr,
    n_elements, hidden_size,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr = 1e-05
):
    # Each program handles a 2D tile of data
    pid = tl.program_id(0)
    n_cols = tl.cdiv(n_elements, hidden_size)
    
    # Determine which elements this program handles
    col_start = pid * BLOCK_SIZE
    col_end = min((pid + 1) * BLOCK_SIZE, n_cols)
    
    if col_start >= n_cols:
        return
    
    # Load weight and bias
    weight = tl.load(weight_ptr + pid * hidden_size)
    bias = tl.load(bias_ptr + pid)
    
    ln_weight = tl.load(ln_weight_ptr + pid)
    ln_bias = tl.load(ln_bias_ptr + pid)
    
    # Process each column
    for col in range(col_start, col_end):
        col_base = col * hidden_size
        
        # Compute mean and variance for the column
        sum_val = 0.0
        sum_sq = 0.0
        
        for i in range(hidden_size):
            idx = col_base + i
            x = tl.load(x_ptr + idx)
            sum_val += x
            sum_sq += x * x
        
        mean = sum_val / hidden_size
        var = (sum_sq / hidden_size) - (mean * mean)
        std = tl.sqrt(var + eps)
        
        # Process each element in the column
        for i in range(hidden_size):
            idx = col_base + i
            
            # Linear + Layer norm
            x = tl.load(x_ptr + idx)
            linear_out = x * weight + bias
            normalized = (linear_out - mean) / std * ln_weight + ln_bias
            
            # Sigmoid
            sigmoid_out = 1.0 / (1.0 + tl.exp(-normalized))
            
            # Store results
            tl.store(out_ptr + idx, normalized)
            tl.store(sigmoid_out_ptr + idx, sigmoid_out)

@torch.fx.wrap
def fused_linear_layer_norm_sigmoid(in_x, in_weight, in_bias, ln_weight, ln_bias):
    # Get tensor shapes
    n_elements = in_x.numel()
    hidden_size = in_x.shape[-1]
    
    # Determine grid size
    BLOCK_SIZE = 64  # Number of columns per program
    n_cols = tl.cdiv(n_elements, hidden_size)
    num_programs = triton.cdiv(n_cols, BLOCK_SIZE)
    
    # Allocate output tensors
    ln_out = torch.empty_like(in_x)
    sigmoid_out = torch.empty_like(in_x)
    
    # Launch kernel
    linear_layer_norm_sigmoid_kernel[(num_programs,)](
        x_ptr=in_x,
        weight_ptr=in_weight,
        bias_ptr=in_bias,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        out_ptr=ln_out,
        sigmoid_out_ptr=sigmoid_out,
        n_elements=n_elements,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05
    )
    
    return ln_out, sigmoid_out

def replacement_func():
    return fused_linear_layer_norm_sigmoid