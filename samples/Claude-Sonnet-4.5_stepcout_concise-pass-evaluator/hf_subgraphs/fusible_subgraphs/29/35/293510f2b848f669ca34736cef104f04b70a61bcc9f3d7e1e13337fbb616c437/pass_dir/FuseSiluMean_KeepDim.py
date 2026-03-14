import torch
import triton
import triton.language as tl

# Pattern matching function for silu + mean with keepdim=True
def pattern(in_0):
    """
    Match: silu(inplace=True) -> mean((2, 3), keepdim=True) -> return (silu, mean)
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def silu_mean_keepdim_kernel(
    input_ptr,
    output_silu_ptr,
    output_mean_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Apply SiLU and compute spatial mean with keepdim
    Each program handles one (batch, channel) pair
    """
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Base offset for this (batch, channel) slice
    base_offset = batch_idx * channels * spatial_size + channel_idx * spatial_size
    
    # Accumulate sum for mean calculation
    accumulator = 0.0
    
    # Process spatial elements in blocks
    for block_start in range(0, spatial_size, BLOCK_SIZE):
        offsets = base_offset + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < spatial_size
        
        # Load input
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Apply SiLU: x * sigmoid(x)
        sigmoid_x = tl.sigmoid(x)
        silu_x = x * sigmoid_x
        
        # Store SiLU result (inplace)
        tl.store(output_silu_ptr + offsets, silu_x, mask=mask)
        
        # Accumulate for mean
        accumulator += tl.sum(tl.where(mask, silu_x, 0.0))
    
    # Compute mean and store (keepdim=True: shape is [B, C, 1, 1])
    mean_val = accumulator / spatial_size
    mean_offset = batch_idx * channels + channel_idx
    tl.store(output_mean_ptr + mean_offset, mean_val)

@torch.fx.wrap
def fused_silu_mean_keepdim(x):
    """
    Fused SiLU + Mean operation (keepdim=True)
    Input shape: [B, C, H, W]
    Output shapes: silu [B, C, H, W], mean [B, C, 1, 1]
    """
    B, C, H, W = x.shape
    spatial_size = H * W
    
    # Output tensors
    out_silu = torch.empty_like(x)
    out_mean = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = (B * C,)
    BLOCK_SIZE = triton.next_power_of_2(min(spatial_size, 1024))
    
    silu_mean_keepdim_kernel[grid](
        x,
        out_silu,
        out_mean,
        B,
        C,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_silu, out_mean)

def replacement_func():
    return fused_silu_mean_keepdim