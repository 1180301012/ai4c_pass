import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches: 4 identical max_pool2d operations on same input, then concat with original
# Note: Must match EXACT op patterns (positional args, no cleanup)
def pattern(tmp_0):
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, False, False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, False, False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, False, False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_1, tmp_2, tmp_3, tmp_4

def replacement_args(tmp_0):
    # Extract necessary inputs (just tmp_0)
    return (tmp_0,)


# Triton kernel for max_pool2d with kernel=5, stride=1, padding=2
@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_h, input_w,
    output_h, output_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Block index for spatial dimensions
    block_h = tl.program_id(0)
    block_w = tl.program_id(1)
    c = tl.program_id(2)

    # Process block of output
    for h in range(block_h * BLOCK_H, min(output_h, (block_h + 1) * BLOCK_H)):
        for w in range(block_w * BLOCK_W, min(output_w, (block_w + 1) * BLOCK_W)):
            # Calculate the 5x5 window starting point (with padding)
            h_start = h * 1 - 2
            w_start = w * 1 - 2
            max_val = tl.float32(-1e9)
            
            # Calculate max over 5x5 window
            for dh in range(5):
                for dw in range(5):
                    h_idx = h_start + dh
                    w_idx = w_start + dw
                    # Boundary check
                    if h_idx >= 0 and h_idx < input_h and w_idx >= 0 and w_idx < input_w:
                        idx = c * input_h * input_w + h_idx * input_w + w_idx
                        val = tl.load(input_ptr + idx)
                        if val > max_val:
                            max_val = val
            # Store result
            out_idx = c * output_h * output_w + h * output_w + w
            tl.store(output_ptr + out_idx, max_val)


# Kernel wrapper (must be torch.fx.wrap)
@torch.fx.wrap
def max_pool2d_wrapper(tmp_0):
    # Get shapes
    batch_size, channels, input_h, input_w = tmp_0.shape
    # Calculate output spatial size (with padding=2, kernel=5, stride=1)
    output_h = (input_h - 5 + 2 * 2) // 1 + 1  # = input_h (20->20)
    output_w = (input_w - 5 + 2 * 2) // 1 + 1  # = input_w (20->20)
    
    # Allocate output tensor
    pooled = torch.empty_like(tmp_0)
    
    # Configure grid for spatial dimensions (small blocks)
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 32
    
    num_blocks_h = (output_h + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (output_w + BLOCK_W - 1) // BLOCK_W
    num_blocks_c = (channels + BLOCK_C - 1) // BLOCK_C
    
    # Launch kernel
    grid = (num_blocks_h, num_blocks_w, num_blocks_c)
    max_pool2d_kernel[grid](
        tmp_0, pooled,
        batch_size, channels, input_h, input_w,
        output_h, output_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return pooled


# Optimized replacement function
@torch.fx.wrap
def optimized_kernel_wrapper(tmp_0):
    # Compute pooled output once
    pooled = max_pool2d_wrapper(tmp_0)
    # Repeat pooled tensor 3 times along channel dimension (equivalent to 3 copies)
    repeated_pooled = torch.cat([pooled, pooled, pooled], dim=1)
    # Concatenate original input with repeated pooled output
    tmp_4 = torch.cat([tmp_0, repeated_pooled], dim=1)
    return tmp_4

def replacement_func():
    return optimized_kernel_wrapper