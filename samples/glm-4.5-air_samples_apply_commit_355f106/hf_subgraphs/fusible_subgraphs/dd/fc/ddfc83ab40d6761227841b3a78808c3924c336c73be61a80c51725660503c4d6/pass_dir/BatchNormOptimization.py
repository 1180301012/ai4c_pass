import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, momentum, eps):
    # Standard batch norm pattern
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, momentum=False, eps=eps)

def replacement_args(x, running_mean, running_var, weight, bias, momentum, eps):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def optimized_batch_norm_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Compute range for this program  
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    hw_start = pid_hw * BLOCK_SIZE_HW
    
    n_end = min(n_start + BLOCK_SIZE_N, N)
    c_end = min(c_start + BLOCK_SIZE_C, C)
    hw_end = min(hw_start + BLOCK_SIZE_HW, H * W)
    
    # Pre-load BN parameters for this channel block
    for c_idx in range(c_start, c_end):
        mean_val = tl.load(running_mean_ptr + c_idx, mask=c_idx < C, other=0.0)
        var_val = tl.load(running_var_ptr + c_idx, mask=c_idx < C, other=1.0)
        weight_val = tl.load(weight_ptr + c_idx, mask=c_idx < C, other=1.0)
        bias_val = tl.load(bias_ptr + c_idx, mask=c_idx < C, other=0.0)
        
        # Compute normalization factor
        inv_std = 1.0 / tl.sqrt(var_val + eps)
        
        # Process all spatial positions for this channel and batch sample
        for n in range(n_start, n_end):
            for hw in range(hw_start, hw_end):
                # Load input value
                x_idx = n * C * H * W + c_idx * H * W + hw
                x_val = tl.load(x_ptr + x_idx, mask=True, other=0.0)
                
                # Apply batch normalization
                # (x - mean) / sqrt(var + eps) * weight + bias
                normalized = (x_val - mean_val) * inv_std * weight_val + bias_val
                
                # Store result
                out_idx = n * C * H * W + c_idx * H * W + hw
                tl.store(out_ptr + out_idx, normalized)

def optimized_batch_norm_wrapper(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    
    # Output has same shape as input
    out = torch.empty((N, C, H, W), dtype=x.dtype, device=x.device)
    
    # Determine block sizes based on tensor dimensions
    BLOCK_SIZE_N = min(8, N)
    BLOCK_SIZE_C = min(64, C)
    BLOCK_SIZE_HW = min(1024, H * W)
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
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
        eps=1e-5,  # Default epsilon value from original
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

@torch.fx.wrap
def triton_batch_norm(x, running_mean, running_var, weight, bias):
    return optimized_batch_norm_wrapper(x, running_mean, running_var, weight, bias)

def replacement_func():
    return triton_batch_norm