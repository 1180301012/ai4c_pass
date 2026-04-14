import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    # Full pattern: addition + mean + identity dropouts + batch norm
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
def fused_add_mean_bn_kernel(
    x_ptr,
    y_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    bn_out_ptr,
    mean_out_ptr,
    n_channels: tl.constexpr,
    n_spatial: tl.constexpr,
    eps: tl.constexpr,
):
    """Fused kernel for addition + mean reduction + batch norm"""
    pid = tl.program_id(0)
    channel_idx = pid
    
    if channel_idx >= n_channels:
        return
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Load and add input data for all spatial positions
    spatial_sum = 0.0
    valid_count = 0
    
    spatial_offsets = tl.arange(0, n_spatial)
    for batch_idx in range(32):  # Assuming max batch size of 32
        batch_offset = batch_idx * n_channels * n_spatial
        channel_offset = channel_idx * n_spatial
        
        # Load and add x and y for this batch and channel
        x_data = tl.load(x_ptr + batch_offset + channel_offset + spatial_offsets)
        y_data = tl.load(y_ptr + batch_offset + channel_offset + spatial_offsets)
        spatial_sum += tl.sum(x_data + y_data)
        valid_count += n_spatial
    
    # Compute mean
    mean_val = spatial_sum / valid_count if valid_count > 0 else 0.0
    
    # Apply batch normalization to the mean value
    inv_std = tl.rsqrt(running_var + eps)
    normalized = (mean_val - running_mean) * inv_std
    bn_output = normalized * weight + bias
    
    # Store results - mean_out_ptr stores per-batch, per-channel means
    # For simplicity, we store the mean for the first batch
    channel_offset = channel_idx
    tl.store(bn_out_ptr + channel_offset, bn_output)
    tl.store(mean_out_ptr + channel_offset, mean_val)

@torch.fx.wrap
def fused_add_mean_bn(x, y, running_mean, running_var, weight, bias):
    """Fused operation: add + mean reduction + batch normalization"""
    N, C, H, W = x.shape
    n_spatial = H * W
    
    # Create output tensors
    bn_output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    mean_output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    eps = 1e-05
    num_programs = C
    
    # Process each sample separately to maintain NxC output dimension
    for i in range(N):
        fused_add_mean_bn_kernel[(num_programs,)](
            x_ptr=x[i],
            y_ptr=y[i],
            running_mean_ptr=running_mean,
            running_var_ptr=running_var,
            weight_ptr=weight,
            bias_ptr=bias,
            bn_out_ptr=bn_output[i],
            mean_out_ptr=mean_output[i],
            n_channels=C,
            n_spatial=n_spatial,
            eps=eps,
        )
    
    return bn_output, mean_output

def replacement_func():
    return fused_add_mean_bn