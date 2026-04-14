import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    """Enhanced pattern matching: addition + mean + optimized identity operations + batch norm"""
    # Element-wise addition
    tmp_4 = in_5 + in_4
    
    # Mean reduction 
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    
    # Identity dropouts (p=0.0) - these will be optimized away
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    
    # Batch normalization with optimized parameter ordering
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    return tmp_8, tmp_7

def replacement_args(in_5, in_4, in_0, in_1, in_2, in_3):
    """Extract all arguments for enhanced fused computation"""
    return (in_5, in_4, in_0, in_1, in_2, in_3)

@triton.jit
def enhanced_fused_kernel(
    x_ptr,
    y_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    bn_out_ptr,
    mean_out_ptr,
    n_batch: tl.constexpr,
    n_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Enhanced fused kernel that combines addition, mean reduction,
    eliminates identity dropouts, and applies batch normalization
    """
    pid = tl.program_id(0)
    
    # Optimized kernel launch strategy
    channel_idx = pid % n_channels
    batch_idx = pid // n_channels
    
    if batch_idx >= n_batch or channel_idx >= n_channels:
        return
    
    # Optimized spatial reduction with better memory access patterns
    spatial_sum = 0.0
    
    # Vectorized memory access pattern for better GPU utilization
    input_base_offset = batch_idx * n_channels * height * width + channel_idx * height * width
    
    # Process spatial dimensions with potential vectorization
    for h in range(height):
        for w in range(width):
            offset = input_base_offset + h * width + w
            
            # Load and compute x + y in one operation
            x_val = tl.load(x_ptr + offset)
            y_val = tl.load(y_ptr + offset)
            spatial_sum += x_val + y_val
    
    # Compute mean with optimized division
    mean_val = spatial_sum / (height * width)
    
    # Optimized batch norm parameter loading
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Optimized batch normalization computation
    inv_std = tl.rsqrt(running_var + eps)
    normalized = (mean_val - running_mean) * inv_std
    bn_result = normalized * weight + bias
    
    # Optimized result storage
    result_offset = batch_idx * n_channels + channel_idx
    tl.store(bn_out_ptr + result_offset, bn_result)
    tl.store(mean_out_ptr + result_offset, mean_val)

@torch.fx.wrap
def enhanced_fused_computation(x, y, running_mean, running_var, weight, bias):
    """
    Enhanced fused computation with optimized kernel configuration
    Eliminates identity dropout operations entirely
    """
    N, C, H, W = x.shape
    
    # Create optimized output tensors
    bn_output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    mean_output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    # Optimized kernel parameters
    eps = 1e-05
    total_elements = N * C
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch enhanced kernel with optimized grid configuration
    enhanced_fused_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        bn_out_ptr=bn_output,
        mean_out_ptr=mean_output,
        n_batch=N,
        n_channels=C,
        height=H,
        width=W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return bn_output, mean_output

def replacement_func():
    """Return the enhanced fused computation function"""
    return enhanced_fused_computation