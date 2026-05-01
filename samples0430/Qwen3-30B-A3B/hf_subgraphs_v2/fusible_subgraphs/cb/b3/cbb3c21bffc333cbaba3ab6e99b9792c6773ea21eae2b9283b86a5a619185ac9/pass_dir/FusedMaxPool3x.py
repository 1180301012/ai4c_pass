import torch
import triton
import triton.language as tl

@triton.jit
def max_pool_kernel_1c(
    input_ptr,
    output_ptr,
    N: tl.int32,
    H: tl.int32,
    W: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    padding: tl.int32,
    kernel_size: tl.int32,
    stride: tl.int32,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Block indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Block offsets
    block_start_h = pid_h * BLOCK_H
    block_start_w = pid_w * BLOCK_W
    
    # Offsets within block
    offs_h = block_start_h + tl.arange(0, BLOCK_H)
    offs_w = block_start_w + tl.arange(0, BLOCK_W)
    
    # Mask for valid output elements
    mask_h = offs_h < H_out
    mask_w = offs_w < W_out
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Calculate input window starts
    input_start_h = block_start_h * stride - padding
    input_start_w = block_start_w * stride - padding

    for i in range(BLOCK_H):
        for j in range(BLOCK_W):
            if not mask[i, j]:
                continue
            h_out = offs_h[i]
            w_out = offs_w[j]
            
            # Calculate input region
            h_start = h_out * stride - padding
            h_end = h_start + kernel_size
            w_start = w_out * stride - padding
            w_end = w_start + kernel_size
            
            # Initialize with a very small value
            max_val = -1e9
            
            # Compute max over the 5x5 window for all batches
            for b in range(N):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        ih = h_start + kh
                        iw = w_start + kw
                        if ih < 0 or ih >= H or iw < 0 or iw >= W:
                            continue
                        input_idx = b * (H * W) + ih * W + iw
                        val = tl.load(input_ptr + input_idx)
                        max_val = max(max_val, val)
            
            output_idx = b * (H_out * W_out) + h_out * W_out + w_out
            tl.store(output_ptr + output_idx, max_val)

@torch.fx.wrap
def max_pool_kernel_wrapper(input_tensor):
    # Extract dimensions
    N, C, H, W = input_tensor.shape
    kernel_size = 5
    stride = 1
    padding = 2
    H_out = (H - kernel_size + 2 * padding) // stride + 1
    W_out = (W - kernel_size + 2 * padding) // stride + 1
    
    # Create output tensor
    output = torch.empty((N, C, H_out, W_out), 
                         dtype=input_tensor.dtype, 
                         device=input_tensor.device)

    # Process each channel independently
    for c in range(C):
        # Extract current channel (as [N, H, W])
        input_chan = input_tensor[:, c:c+1, :, :].contiguous().view(N, H, W)
        output_chan = output[:, c:c+1, :, :].contiguous().view(N, H_out, W_out)
        
        # Compute grid size
        grid_h = (H_out + 15) // 16
        grid_w = (W_out + 15) // 16
        
        # Launch kernel
        max_pool_kernel_1c[(grid_h, grid_w)](
            input_chan,
            output_chan,
            N,
            H,
            W,
            H_out,
            W_out,
            padding,
            kernel_size,
            stride,
            16,
            16
        )
    
    return output

# Pattern matching function - must mirror model.py exactly
# IMPORTANT: Do not include cleanup statements (tmp_x = None)
def pattern(tmp_0):
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return tmp_0, tmp_1, tmp_2, tmp_3

def replacement_args(tmp_0, tmp_1, tmp_2, tmp_3):
    return (tmp_0,)

def replacement_func():
    return max_pool_kernel_wrapper