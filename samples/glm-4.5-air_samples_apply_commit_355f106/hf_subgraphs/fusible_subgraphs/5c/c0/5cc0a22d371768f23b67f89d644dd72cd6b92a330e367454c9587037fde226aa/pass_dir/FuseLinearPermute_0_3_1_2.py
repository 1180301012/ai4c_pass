import torch
import triton
import triton.language as tl

# Pattern matching function: linear operation only
def pattern(x, weight, bias):
    # Linear operation: y = x @ weight.T + bias
    linear_out = torch.nn.functional.linear(x, weight, bias)
    return linear_out

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Simple Triton kernel that avoids problematic operations
@triton.jit
def linear_kernel_simple(
    x_ptr, weight_ptr, bias_ptr, 
    out_ptr,
    N, H, W, C_in, C_out,
):
    # Use only 2D grid to avoid complexity
    pid_m = tl.program_id(0)  # Output channels  
    pid_pos = tl.program_id(1)  # Spatial position (flattened H*W)
    
    # Check bounds
    total_positions = N * H * W
    if pid_pos >= total_positions or pid_m >= C_out:
        return
    
    # Calculate actual coordinates
    batch_idx = pid_pos // (H * W)
    local_h = (pid_pos // W) % H
    local_w = pid_pos % W
    
    # Calculate output offset
    out_offset = batch_idx * (H * W * C_out) + local_h * (W * C_out) + local_w * C_out + pid_m
    
    # Load bias
    bias_val = tl.load(bias_ptr + pid_m)
    
    # Simple scalar approach for small C_in (only 3 channels)
    result = bias_val
    for k in range(C_in):
        weight_offset = pid_m * C_in + k
        x_offset = batch_idx * (H * W * C_in) + local_h * (W * C_in) + local_w * C_in + k
        
        # Load individual values
        weight_val = tl.load(weight_ptr + weight_offset)
        x_val = tl.load(x_ptr + x_offset)
        
        result += weight_val * x_val
    
    # Store result
    tl.store(out_ptr + out_offset, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_linear(x, weight, bias):
    # Input x shape: [N, H, W, C_in] -> [1, 196, 196, 3]
    # Weight shape: [C_out, C_in] -> [16, 3]
    # Bias shape: [C_out] -> [16]
    
    N, H, W, C_in = x.shape
    C_out = weight.shape[0]
    
    # Output shape: [N, H, W, C_out]
    output = torch.empty((N, H, W, C_out), device=x.device, dtype=x.dtype)
    
    # Use 2D grid: (C_out, N * H * W)
    grid = (C_out, N * H * W)
    
    # Launch kernel
    linear_kernel_simple[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        N=N, H=H, W=W, C_in=C_in, C_out=C_out,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_linear