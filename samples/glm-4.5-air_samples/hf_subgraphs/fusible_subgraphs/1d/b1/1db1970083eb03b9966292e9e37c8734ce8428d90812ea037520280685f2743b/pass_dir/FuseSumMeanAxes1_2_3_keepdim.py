import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = x.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1

def replacement_args(x):
    return (x,)

@triton.jit
def fused_sum_mean_kernel(
    x_ptr,
    out_ptr,
    N,  # batch size
    C,  # channels (size 2 originally)
    F,  # features (256) 
    H,  # height 
    W,  # width
    BLOCK_SIZE_H: tl.constexpr,
):
    # Each program handles one element in the output tensor
    # Output shape: [N, 1, 1, 1, W] -> program_id = (batch, width)
    batch_idx = tl.program_id(0)
    width_idx = tl.program_id(1)
    
    # Initialize accumulator for this position
    acc = 0.0
    
    # Create coordinate offsets within the block
    h_off = tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_off < H
    
    # Iterate over all features and channels
    for c in range(C):  # C = 2, hardcoded
        for f in range(F):
            # Calculate base offset for this (batch, channel, feature, width)
            base_offset = batch_idx * (C * F * H * W) + c * (F * H * W) + f * (H * W) + width_idx
            
            # Load all heights for this combination
            offsets = base_offset + h_off
            
            # Load data and accumulate
            x_val = tl.load(x_ptr + offsets, mask=h_mask, other=0.0)
            acc += tl.sum(x_val)
    
    # Compute final mean
    acc = acc / (C * F * H)
    
    # Store result
    out_offset = batch_idx * (1 * 1 * 1 * W) + width_idx
    tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def fused_sum_mean_wrapper(x):
    # Get input dimensions
    N, C, F, H, W = x.shape
    
    # Create output tensor - final shape should be [N, 1, 1, 1, W]
    out_shape = [N, 1, 1, 1, W]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Use appropriate block size for height
    BLOCK_SIZE_H = min(32, H)
    if H <= 16:
        BLOCK_SIZE_H = H
    
    # Calculate grid dimensions: [batch_size, width]
    grid = (N, W)
    
    # Launch kernel
    fused_sum_mean_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        F=F,
        H=H,
        W=W,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return out

def replacement_func():
    return fused_sum_mean_wrapper