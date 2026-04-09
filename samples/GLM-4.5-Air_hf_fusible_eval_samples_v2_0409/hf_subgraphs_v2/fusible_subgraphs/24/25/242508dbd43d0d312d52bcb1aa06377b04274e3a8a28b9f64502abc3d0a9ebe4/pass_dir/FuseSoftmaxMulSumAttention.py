import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Pattern matching for fused softmax multiply sum attention operation"""
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_attention_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_batch,
    n_groups,
    n_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel for fused softmax * multiply * sum attention
    Process: softmax along groups, multiply, then sum along groups
    """
    pid = tl.program_id(0)
    
    # Convert to 2D grid: (batch*spatial, groups*channels)
    n_spatial = height * width
    grid_spatial = n_batch * n_spatial
    grid_groups_channels = n_groups * n_channels
    
    pid_spatial = pid // grid_groups_channels  # which spatial position
    pid_groups_channels = pid % grid_groups_channels  # which group and channel
    
    pid_batch = pid_spatial // n_spatial
    pid_spatial_local = pid_spatial % n_spatial
    pid_group = pid_groups_channels // n_channels
    pid_channel = pid_groups_channels % n_channels
    
    # Return if out of bounds
    if pid_batch >= n_batch:
        return
    if pid_group >= n_groups:
        return
    if pid_channel >= n_channels:
        return
    
    # Load attention weights for this batch and group (all channels)
    weights = tl.load(in_1_ptr + pid_batch * n_groups * n_channels + pid_group * n_channels + pid_channel)
    
    # Apply softmax across groups for this channel
    # Load all group weights for this channel and batch
    max_weight = -10000.0
    for g in range(n_groups):
        w = tl.load(in_1_ptr + pid_batch * n_groups * n_channels + g * n_channels + pid_channel)
        max_weight = tl.maximum(max_weight, w)
    
    # Compute exponentials and sum
    sum_exp = 0.0
    for g in range(n_groups):
        w = tl.load(in_1_ptr + pid_batch * n_groups * n_channels + g * n_channels + pid_channel)
        sum_exp += tl.exp(w - max_weight)
    
    # Compute weighted sum across groups for this channel and spatial position
    weighted_sum = 0.0
    for g in range(n_groups):
        # Load feature for this group, channel, and spatial position
        feat_idx = (pid_batch * n_groups * n_channels * n_spatial + 
                   g * n_channels * n_spatial + 
                   pid_channel * n_spatial + 
                   pid_spatial_local)
        feature = tl.load(in_0_ptr + feat_idx)
        
        # Load weight, compute softmax, and multiply
        w = tl.load(in_1_ptr + pid_batch * n_groups * n_channels + g * n_channels + pid_channel)
        softmax_weight = tl.exp(w - max_weight) / (sum_exp + 1e-6)
        weighted_sum += feature * softmax_weight
    
    # Store output: [batch, channels, height, width] 
    out_idx = pid_batch * n_channels * n_spatial + pid_channel * n_spatial + pid_spatial_local
    tl.store(out_ptr + out_idx, weighted_sum)

@torch.fx.wrap
def fused_attention_forward(in_0, in_1):
    """
    Forward pass for fused softmax multiply sum attention
    in_0: [batch_size, groups, channels, height, width] - feature maps
    in_1: [batch_size, groups, channels, 1, 1] - attention weights
    out: [batch_size, channels, height, width] - result
    """
    batch_size, groups, channels, height, width = in_0.shape
    
    # Get device from input tensors
    device = in_0.device
    
    # Create output tensor - [batch_size, channels, height, width]
    out_shape = [batch_size, channels, height, width]
    out = torch.empty(out_shape, dtype=in_0.dtype, device=device)
    
    # Configure block sizes
    BLOCK_SIZE_M = 1  # each program handles 1 channel group
    BLOCK_SIZE_N = 1  # each program handles 1 spatial position
    
    # Calculate grid size
    n_spatial = height * width
    grid_spatial = batch_size * n_spatial  # total spatial positions across all batches
    grid_groups_channels = groups * channels  # total groups * channels
    
    total_grid_size = grid_spatial * grid_groups_channels
    
    # Launch kernel
    grid = (total_grid_size,)
    fused_attention_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_batch=batch_size,
        n_groups=groups,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_attention_forward