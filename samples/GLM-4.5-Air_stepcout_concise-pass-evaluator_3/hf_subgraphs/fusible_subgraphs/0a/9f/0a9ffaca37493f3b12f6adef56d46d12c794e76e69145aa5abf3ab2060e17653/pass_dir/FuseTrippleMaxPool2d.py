import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_maxpool_kernel(
    x_ptr, 
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Maxpool parameters: kernel_size=5, stride=1, padding=2
    kernel_size = 5
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 2, 2
    
    # Compute grid position
    batch_idx = tl.program_id(0)
    channel_in = tl.program_id(1)
    
    # Adjust for batch dimension
    x_ptr += batch_idx * C * H * W
    out_ptr += batch_idx * (4 * C) * H * W
    
    # Calculate output spatial dimensions
    out_h = (H + 2 * pad_h - kernel_size) // stride_h + 1
    out_w = (W + 2 * pad_w - kernel_size) // stride_w + 1
    
    # Handle the original ReLU output (no pooling)
    if channel_in < C:
        # Copy original channels to output
        for h in range(0, H, BLOCK_SIZE_M):
            for w in range(0, W, BLOCK_SIZE_N):
                offsets_h = h + tl.arange(0, BLOCK_SIZE_M)
                offsets_w = w + tl.arange(0, BLOCK_SIZE_N)
                mask_h = offsets_h < H
                mask_w = offsets_w < W
                mask = mask_h[:, None] & mask_w[None, :]
                
                offsets = offsets_h[:, None] * W + offsets_w[None, :]
                x_val = tl.load(x_ptr + channel_in * H * W + offsets, mask=mask, other=0.0)
                tl.store(out_ptr + channel_in * H * W + offsets, x_val, mask=mask)
    
    # Handle the three max-pooling outputs
    for pool_idx in range(3):
        pool_channel_out = C + pool_idx * C
        if channel_in < C:
            # For each position in output spatial dimensions
            for hh in range(0, out_h, BLOCK_SIZE_M):
                for ww in range(0, out_w, BLOCK_SIZE_N):
                    # Output coordinates
                    out_offsets_h = hh + tl.arange(0, BLOCK_SIZE_M)
                    out_offsets_w = ww + tl.arange(0, BLOCK_SIZE_N)
                    out_mask_h = out_offsets_h < out_h
                    out_mask_w = out_offsets_w < out_w
                    out_mask = out_mask_h[:, None] & out_mask_w[None, :]
                    
                    # Calculate input window coordinates
                    in_start_h = out_offsets_h * stride_h - pad_h
                    in_start_w = out_offsets_w * stride_w - pad_w
                    
                    # Initialize max values
                    max_val = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('-inf'), dtype=tl.float32)
                    
                    # Sliding window over kernel
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            # Current input coordinates
                            current_h = in_start_h + kh
                            current_w = in_start_w + kw
                            
                            # Check bounds
                            h_mask = current_h >= 0 & current_h < H
                            w_mask = current_w >= 0 & current_w < W
                            window_mask = h_mask[:, None] & w_mask[None, :]
                            
                            # Only process valid coordinates
                            if tl.any(window_mask):
                                # Calculate input offsets relative to current window position
                                local_offsets = (current_h[:, None] - in_start_h[:, None]) * W + (current_w[None, :] - in_start_w[None, :])
                                
                                # Load input values
                                x_val = tl.load(x_ptr + channel_in * H * W + local_offsets, mask=window_mask, other=0.0)
                                
                                # Update max values
                                max_val = tl.maximum(max_val, x_val)
                    
                    # Store the max values to output
                    out_offsets = out_offsets_h[:, None] * out_w + out_offsets_w[None, :]
                    tl.store(out_ptr + pool_channel_out * H * W + out_offsets, max_val, mask=out_mask)

@torch.fx.wrap
def fused_maxpool_forward(x):
    # Get input dimensions
    N, C, H, W = x.shape
    
    # Prepare output tensor
    output = torch.empty((N, 4 * C, H, W), dtype=x.dtype, device=x.device)
    
    # Launch Triton kernel
    # Use conservative block sizes for safety
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 1
    
    grid = (
        N,
        4 * C,
    )
    
    fused_maxpool_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_maxpool_forward