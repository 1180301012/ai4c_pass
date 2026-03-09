import torch
import triton
import triton.language as tl
import math

def pattern(x, running_mean, running_var, weight, bias):
    # Batch norm
    y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    # Spatial mean
    mean = y.mean((2, 3), keepdim=True)
    return y, mean

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def fused_batch_norm_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    N,  # batch size
    C,  # channels  
    H,  # height
    W,  # width
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute spatial size
    spatial_size = H * W
    
    # Initialize spatial accumulator
    spatial_acc = 0.0
    
    # Load BN parameters
    mean_val = tl.load(running_mean_ptr + pid_c)
    var_val = tl.load(running_var_ptr + pid_c)
    weight_val = tl.load(weight_ptr + pid_c)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Compute 1/sqrt(var + eps)
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Process spatial dimensions
    for h in range(0, H, 1):  # Process each spatial location
        for w in range(0, W, 1):  # In practice, we'd tile better, but for simplicity
            ptr = pid_n * C * H * W + pid_c * H * W + h * W + w
            x_val = tl.load(x_ptr + ptr, other=0.0)
            
            # Batch norm formula: y = (x - mean) * weight * rstd + bias
            y_val = (x_val - mean_val) * weight_val * rstd + bias_val
            
            # Accumulate for spatial mean
            spatial_acc += y_val
    
    # Compute spatial mean
    spatial_mean = spatial_acc / spatial_size
    
    # Store mean (to shared memory or global depending on optimization)
    tl.store(mean_ptr + pid_n * C + pid_c, spatial_mean)

@torch.fx.wrap
def fused_batch_norm_mean(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    
    # Allocate outputs
    out = torch.empty_like(x)
    mean_out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Calculate grid size
    grid = (N, C)
    
    # Launch kernel
    fused_batch_norm_mean_kernel[grid](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        mean_ptr=mean_out,
        N=N,
        C=C,
        H=H,
        W=W,
       eps=1e-05,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_C=1,
    )
    
    return out, mean_out

def replacement_func():
    return fused_batch_norm_mean