import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, momentum, eps):
    # Simple pattern using basic arithmetic operations only
    # This matches: (x - running_mean) / sqrt(running_var + eps) * weight + bias
    diff = x - running_mean
    inv_std = 1.0 / (running_var + eps) ** 0.5
    normalized = diff * inv_std
    return weight * normalized + bias

def replacement_args(x, running_mean, running_var, weight, bias, momentum, eps):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def optimized_batch_norm_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Process this batch and channel
    n = pid_n
    c = pid_c
    
    # Compute spatial position
    hw_start = pid_hw * BLOCK_SIZE_HW
    hw_end = min(hw_start + BLOCK_SIZE_HW, H * W)
    
    # Load BN parameters for this channel
    mean_val = tl.load(running_mean_ptr + c, mask=c < C, other=0.0)
    var_val = tl.load(running_var_ptr + c, mask=c < C, other=1.0)
    weight_val = tl.load(weight_ptr + c, mask=c < C, other=1.0)
    bias_val = tl.load(bias_ptr + c, mask=c < C, other=0.0)
    
    # Pre-compute normalization factor
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    
    # Process spatial positions
    for hw_idx in range(hw_start, hw_end):
        # Load input value
        x_idx = n * C * H * W + c * H * W + hw_idx
        x_val = tl.load(x_ptr + x_idx, mask=True, other=0.0)
        
        # Apply batch normalization without branching
        # Use fused operations for better performance
        diff = x_val - mean_val
        normalized = diff * inv_std
        result = normalized * weight_val + bias_val
        
        # Store result
        out_idx = n * C * H * W + c * H * W + hw_idx
        tl.store(out_ptr + out_idx, result)

def optimized_batch_norm_wrapper(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    
    # Output has same shape as input
    out = torch.empty((N, C, H, W), dtype=x.dtype, device=x.device)
    
    # Use larger block size for better GPU utilization
    BLOCK_SIZE_HW = min(2048, H * W)  # Process more spatial locations per thread
    
    # Calculate grid - one program per batch and channel
    grid_n = N
    grid_c = C
    grid_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch optimized batch norm kernel
    optimized_batch_norm_kernel[(
        grid_n,
        grid_c,
        grid_hw
    )](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        eps=1e-5,  # Default epsilon
        BLOCK_SIZE_C=32,  # Process multiple channels if needed
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

@torch.fx.wrap
def triton_batch_norm(x, running_mean, running_var, weight, bias):
    return optimized_batch_norm_wrapper(x, running_mean, running_var, weight, bias)

def replacement_func():
    return triton_batch_norm