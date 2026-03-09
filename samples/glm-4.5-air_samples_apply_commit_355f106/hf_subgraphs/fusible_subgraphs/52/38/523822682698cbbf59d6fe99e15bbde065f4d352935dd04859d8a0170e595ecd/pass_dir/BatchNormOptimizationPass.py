import torch
import triton
import triton.language as tl

# Pattern matching function for batch norm operation: torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
def pattern(in_7, in_0, in_1, in_3, in_2):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7

# Argument extraction function
def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

# Optimized batch norm kernel using Triton
@triton.jit
def batch_norm_kernel(
    x_ptr,        # input [batch_size, features]
    mean_ptr,     # running mean [features]
    var_ptr,      # running var [features] 
    weight_ptr,   # weight/scale [features]
    bias_ptr,     # bias/shift [features]
    out_ptr,      # output [batch_size, features]
    batch_size,
    features,
    BLOCK_SIZE_M: tl.constexpr,  # batch dimension block size
    BLOCK_SIZE_N: tl.constexpr,  # features dimension block size
):
    # Program identifiers
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # features dimension
    
    # Compute ranges for this program  
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offset ranges
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < features
    
    # Load normalization parameters
    mean_ptrs = mean_ptr + n_offsets
    var_ptrs = var_ptr + n_offsets  
    weight_ptrs = weight_ptr + n_offsets
    bias_ptrs = bias_ptr + n_offsets
    
    mean = tl.load(mean_ptrs, mask=n_mask, other=0.0)
    var = tl.load(var_ptrs, mask=n_mask, other=1.0)
    weight = tl.load(weight_ptrs, mask=n_mask, other=1.0)
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    
    # Precompute normalization factors (broadcast across batch dimension)
    # output = (x - mean) * weight / sqrt(var + epsilon) + bias
    sqrt_var = tl.sqrt(var + 1e-05)
    inv_std = weight / sqrt_var
    norm_shift = bias - mean * inv_std
    
    # Process batch dimension
    for m in range(0, batch_size, BLOCK_SIZE_M):
        m_off = m + m_offsets
        
        # Load input data: [BLOCK_SIZE_M, BLOCK_SIZE_N]
        x_ptrs = x_ptr + (m_off[:, None] * features + n_offsets[None, :])
        x = tl.load(x_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)
        
        # Apply normalization: y = (x - mean) * inv_std + norm_shift
        y = (x - mean[None, :]) * inv_std[None, :] + norm_shift[None, :]
        
        # Store result
        out_ptrs = out_ptr + (m_off[:, None] * features + n_offsets[None, :])
        tl.store(out_ptrs, y, mask=(m_mask[:, None] & n_mask[None, :]))

# Kernel wrapper
@torch.fx.wrap
def triton_batch_norm(x, running_mean, running_var, weight, bias):
    batch_size, features = x.shape
    
    # Set block sizes based on tensor dimensions
    BLOCK_SIZE_M = 64 if batch_size >= 64 else batch_size
    BLOCK_SIZE_N = 128 if features >= 128 else features
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Create output tensor
    out = torch.empty((batch_size, features), dtype=torch.float32, device=x.device)
    
    # Launch kernel
    batch_norm_kernel[grid](
        x_ptr=x,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        features=features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_batch_norm