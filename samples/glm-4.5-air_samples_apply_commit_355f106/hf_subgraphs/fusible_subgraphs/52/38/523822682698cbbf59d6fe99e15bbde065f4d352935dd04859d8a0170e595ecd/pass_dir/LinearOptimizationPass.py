import torch
import triton
import triton.language as tl

# Pattern matching function for linear operation: torch.nn.functional.linear(in_6, in_5, in_4)
def pattern(in_6, in_5, in_4):
    tmp_6 = torch.nn.functional.linear(in_6, in_5, in_4)
    return tmp_6

# Argument extraction function
def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

# Optimized linear kernel using Triton
@triton.jit
def linear_kernel(
    x_ptr,      # input [batch_size, in_features] 
    w_ptr,      # weight [out_features, in_features]
    b_ptr,      # bias [out_features]
    out_ptr,    # output [batch_size, out_features]
    batch_size,
    in_features,
    out_features,
):
    # Program identifiers
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output features dimension
    
    # Check ranges
    m_offset = pid_m
    n_offset = pid_n
    
    if m_offset >= batch_size or n_offset >= out_features:
        return
    
    # Compute pointer addresses
    x_base = x_ptr + m_offset * in_features
    w_base = w_ptr + n_offset * in_features
    b_base = b_ptr + n_offset
    out_base = out_ptr + m_offset * out_features + n_offset
    
    # Load bias
    bias = tl.load(b_base)
    
    # Compute dot product
    acc = bias
    for k in range(in_features):
        x_val = tl.load(x_base + k)
        w_val = tl.load(w_base + k)
        acc += x_val * w_val
    
    # Store result
    tl.store(out_base, acc)

# Kernel wrapper
@torch.fx.wrap
def triton_linear(x, w, b):
    batch_size, in_features = x.shape
    out_features = b.shape[0]
    
    # Calculate grid dimensions - launch one program per output element
    grid_m = batch_size
    grid_n = out_features
    grid = (grid_m, grid_n)
    
    # Create output tensor
    out = torch.empty((batch_size, out_features), dtype=torch.float32, device=x.device)
    
    # Launch kernel
    linear_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_linear