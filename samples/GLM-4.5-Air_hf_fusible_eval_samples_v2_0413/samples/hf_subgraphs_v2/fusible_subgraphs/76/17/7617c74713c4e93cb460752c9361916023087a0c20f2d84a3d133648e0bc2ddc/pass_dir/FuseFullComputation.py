import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    # Full pattern matching: addition + mean + identity dropouts + batch norm
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_8, tmp_7

def replacement_args(in_5, in_4, in_0, in_1, in_2, in_3):
    # Extract all arguments for fused kernel
    return (in_5, in_4, in_0, in_1, in_2, in_3)

@triton.jit
def fused_compute_kernel(
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
):
    """Fused kernel for entire computation sequence"""
    pid = tl.program_id(0)
    
    # Compute batch and channel indices from program ID
    channel_idx = pid % n_channels
    batch_idx = pid // n_channels
    
    if batch_idx >= n_batch or channel_idx >= n_channels:
        return
    
    # Compute mean addition over spatial dimensions (x + y)
    spatial_sum = 0.0
    spatial_count = 0
    
    input_base_offset = batch_idx * n_channels * height * width + channel_idx * height * width
    
    # Iterate through spatial dimensions to compute mean of (x + y)
    for h in range(height):
        for w in range(width):
            offset = input_base_offset + h * width + w
            
            # Load x and y values and add them
            x_val = tl.load(x_ptr + offset)
            y_val = tl.load(y_ptr + offset)
            spatial_sum += x_val + y_val
            spatial_count += 1
    
    # Compute mean value
    mean_val = spatial_sum / spatial_count if spatial_count > 0 else 0.0
    
    # Load batch normalization parameters
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Apply batch normalization to the mean value
    inv_std = tl.rsqrt(running_var + eps)
    normalized = (mean_val - running_mean) * inv_std
    bn_result = normalized * weight + bias
    
    # Store results - bn_out_ptr stores batch norm result, mean_out_ptr stores intermediate mean
    # We need to store results for both outputs that are returned
    bn_out_offset = batch_idx * n_channels + channel_idx
    mean_out_offset = batch_idx * n_channels + channel_idx
    
    tl.store(bn_out_ptr + bn_out_offset, bn_result)
    tl.store(mean_out_ptr + mean_out_offset, mean_val)

@torch.fx.wrap
def fused_full_computation(x, y, running_mean, running_var, weight, bias):
    """
    Fused computation: 
    - Addition of x and y
    - Mean reduction over spatial dimensions
    - Identity dropout operations (eliminated)
    - Batch normalization
    Returns (batch_norm_result, intermediate_mean)
    """
    N, C, H, W = x.shape
    
    # Create output tensors - first is batch norm result, second is intermediate mean
    bn_output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    mean_output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    eps = 1e-05
    total_elements = N * C
    num_programs = total_elements
    
    # Launch fused kernel
    fused_compute_kernel[(num_programs,)](
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
    )
    
    return bn_output, mean_output

def replacement_func():
    return fused_full_computation