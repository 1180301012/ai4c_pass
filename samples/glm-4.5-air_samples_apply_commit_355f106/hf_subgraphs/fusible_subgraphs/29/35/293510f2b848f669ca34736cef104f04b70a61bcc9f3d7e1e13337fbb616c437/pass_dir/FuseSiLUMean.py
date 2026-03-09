import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.silu(x, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

def replacement_args(x):
    return (x,)

# Optimized fused SiLU + Mean kernel
@triton.jit
def fused_silu_mean_kernel(
    x_ptr,
    out_silu_ptr,
    out_mean_ptr,
    batch_size,      # B
    channels,        # C  
    height,          # H
    width,           # W
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Each program handles one channel (mean reduction per channel)
    c = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE_N
    offsets_x = block_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets_x < (height * width)
    
    # Load input data for this channel
    x_base = x_ptr + c * (height * width) + (batch_size * channels * height * width)
    x = tl.load(x_base + offsets_x, mask=mask, other=0.0)
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    
    # Store SiLU output
    tl.store(out_silu_ptr + c * (height * width) + (batch_size * channels * height * width) + offsets_x, 
             silu_x, mask=mask)
    
    # Compute mean for this channel
    if mask[0]:  # Only compute mean if we have valid elements
        channel_sum = tl.sum(silu_x, axis=0)
        channel_mean = channel_sum / tl.sum(mask)
        tl.store(out_mean_ptr + c + (batch_size * channels), channel_mean)

@triton.jit
def fused_silu_mean_kernel_no_keepdim(
    x_ptr,
    out_silu_ptr,
    out_mean_ptr,
    batch_size,      # B
    channels,        # C  
    height,          # H
    width,           # W
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Each program handles one channel (mean reduction per channel)
    c = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE_N
    offsets_x = block_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets_x < (height * width)
    
    # Load input data for this channel
    x_base = x_ptr + c * (height * width) + (batch_size * channels * height * width)
    x = tl.load(x_base + offsets_x, mask=mask, other=0.0)
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    
    # Store SiLU output
    tl.store(out_silu_ptr + c * (height * width) + (batch_size * channels * height * width) + offsets_x, 
             silu_x, mask=mask)
    
    # Compute mean for this channel
    if mask[0]:  # Only compute mean if we have valid elements
        channel_sum = tl.sum(silu_x, axis=0)
        channel_mean = channel_sum / tl.sum(mask)
        tl.store(out_mean_ptr + c, channel_mean)

@torch.fx.wrap
def fused_silu_mean(x, keepdim=True):
    B, C, H, W = x.shape
    total_elements = B * C * H * W
    
    # Output buffers
    silu_out = torch.empty_like(x, device=x.device)
    
    if keepdim:
        mean_out = torch.empty((B, C), dtype=torch.float32, device=x.device)
        BLOCK_SIZE_M = 256
        BLOCK_SIZE_N = 1024
        grid = (C, (total_elements + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
        
        fused_silu_mean_kernel[grid](
            x, silu_out, mean_out,
            B, C, H, W,
            BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    else:
        mean_out = torch.empty(C, dtype=torch.float32, device=x.device)
        BLOCK_SIZE_M = 256
        BLOCK_SIZE_N = 1024
        grid = (C, (total_elements + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
        
        fused_silu_mean_kernel_no_keepdim[grid](
            x, silu_out, mean_out,
            B, C, H, W,
            BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    
    return silu_out, mean_out

def replacement_func():
    return fused_silu_mean