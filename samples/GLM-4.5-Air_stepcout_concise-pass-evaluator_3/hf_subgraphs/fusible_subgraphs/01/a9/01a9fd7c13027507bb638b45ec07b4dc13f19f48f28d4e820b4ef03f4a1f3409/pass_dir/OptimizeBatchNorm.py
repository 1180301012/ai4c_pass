import torch
import triton
import triton.language as tl

# Pattern matching for batch normalization
def pattern(input_tensor, running_mean, running_var, weight, bias):
    """
    Pattern: BatchNorm operation like tmp_6 = batch_norm(tmp_5, running_mean, running_var, weight, bias, ...)
    """
    # BatchNorm with exact signature from original computation
    result = torch.nn.functional.batch_norm(
        input_tensor, 
        running_mean, 
        running_var, 
        weight, 
        bias, 
        False,  # momentum
        0.1,     # eps
        1e-05    # training
    )
    return result

# Argument extraction function
def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    """
    Extract arguments for batch normalization
    """
    return input_tensor, running_mean, running_var, weight, bias

# Optimized BatchNorm kernel
@triton.jit
def optimized_batchnorm_kernel(
    x_ptr,           # Input tensor pointer
    mean_ptr,        # Running mean pointer  
    var_ptr,         # Running var pointer
    weight_ptr,      # Weight pointer
    bias_ptr,        # Bias pointer
    out_ptr,         # Output tensor pointer
    N,               # Number of channels
    H,               # Height dimension
    W,               # Width dimension  
    C,               # Batch dimension
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Calculate program IDs
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Handle spatial dimensions (H, W)
    h_offset = (pid_hw // ((W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)) * BLOCK_SIZE_HW
    w_offset = (pid_hw % ((W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)) * BLOCK_SIZE_HW
    
    # Handle channel dimension
    n_offset = pid_n * BLOCK_SIZE_N
    c_offset = pid_c
    
    # Load BatchNorm parameters (shared across all spatial positions)
    if n_offset < N:
        mean = tl.load(mean_ptr + n_offset)
        var = tl.load(var_ptr + n_offset)
        weight = tl.load(weight_ptr + n_offset)
        bias = tl.load(bias_ptr + n_offset)
        
        # Calculate sqrt(var + eps) for BatchNorm
        var_plus_eps = var + 1e-05
        inv_std = 1.0 / tl.sqrt(var_plus_eps)
        
        # Process spatial block
        for h in range(h_offset, h_offset + BLOCK_SIZE_HW):
            if h < H:
                for w in range(w_offset, w_offset + BLOCK_SIZE_HW):
                    if w < W:
                        # Load input
                        offset = c_offset * N * H * W + n_offset * H * W + h * W + w
                        x_val = tl.load(x_ptr + offset)
                        
                        # BatchNorm normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
                        norm_val = (x_val - mean) * inv_std * weight + bias
                        
                        # Store result
                        tl.store(out_ptr + offset, norm_val)

@torch.fx.wrap
def optimized_batchnorm(input_tensor, running_mean, running_var, weight, bias):
    """
    Optimized BatchNorm operation using Triton
    """
    # Get tensor dimensions
    C, N, H, W = input_tensor.shape  # Batch, Channels, Height, Width
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Choose block sizes based on typical GPU occupancy
    BLOCK_SIZE_N = min(64, N)  # Block size for channels  
    BLOCK_SIZE_HW = 64         # Block size for spatial dimensions
    
    # Calculate grid sizes
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid_c = C
    
    # Launch kernel
    optimized_batchnorm_kernel[(grid_n, grid_hw, grid_c)](
        x_ptr=input_tensor,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        N=N,
        H=H,
        W=W,
        C=C,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_batchnorm