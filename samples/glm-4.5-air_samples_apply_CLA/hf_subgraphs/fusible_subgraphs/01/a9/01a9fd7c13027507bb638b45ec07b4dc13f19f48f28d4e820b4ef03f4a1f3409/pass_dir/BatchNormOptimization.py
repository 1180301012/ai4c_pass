import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute batch norm parameters based on spatial dimensions
    # For 4D tensor [B, C, H, W], we process per-channel
    stride_B = x_ptr.stride(0) if x_ptr.stride(0) == 1 else 0
    stride_C = x_ptr.stride(1) if x_ptr.stride(1) == x_ptr.size(2) * x_ptr.size(3) else 0
    stride_H = x_ptr.stride(2) if x_ptr.stride(2) == x_ptr.size(3) else 0
    stride_W = x_ptr.stride(3) if x_ptr.stride(3) == 1 else 0
    
    # Handle different tensor shapes
    if stride_B != 0:  # Full tensor processing
        # For each element, determine which channel it belongs to
        total_channels = running_mean_ptr.shape[0]
        channel_size = x.numel() // total_channels
        
        # Load parameters for current channel
        channel_idx = (offsets // channel_size) % total_channels
        mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < total_channels)
        var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < total_channels)
        weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < total_channels) 
        bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < total_channels)
        
        # Batch norm computation
        inv_std = 1.0 / tl.sqrt(var + eps)
        normalized = (x - mean) * inv_std * weight + bias
        tl.store(out_ptr + offsets, normalized, mask=mask)
    else:
        # Simplified case when we can process directly by channels
        channel_idx = offsets % running_mean_ptr.shape[0]
        mean = tl.load(running_mean_ptr + channel_idx)
        var = tl.load(running_var_ptr + channel_idx)
        weight = tl.load(weight_ptr + channel_idx)
        bias = tl.load(bias_ptr + channel_idx)
        
        # Batch norm computation
        inv_std = 1.0 / tl.sqrt(var + eps)
        normalized = (x - mean) * inv_std * weight + bias
        tl.store(out_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap  
def triton_batch_norm(x, running_mean, running_var, weight, bias, eps=1e-05):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    batch_norm_kernel[(num_programs,)](
        x=x,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        out=out,
        n_elements=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(x, running_mean, running_var, weight, bias):
    # Pattern matches batch_norm operation
    tmp_6 = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, 
        training=False, momentum=0.1, eps=1e-05
    )
    return tmp_6

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

def replacement_func():
    return triton_batch_norm