import torch
import triton
import triton.language as tl

# Pattern matching function - matches transpose(-2, -1)
def pattern(k):
    tmp_4 = k.transpose(-2, -1)
    return tmp_4

# Argument extraction function
def replacement_args(k):
    return (k,)

# Optimized kernel for transpose
# Input: [B, C, H, W], Output: [B, C, W, H]
@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    B, C, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Output shape is [B, C, W, H]
    WH = W * H
    CWH = C * WH
    
    b = offsets // CWH
    remainder = offsets % CWH
    c = remainder // WH
    remainder = remainder % WH
    w_out = remainder // H
    h_out = remainder % H
    
    # Input at [b, c, h, w] -> output at [b, c, w, h]
    # Input offset for [b, c, h_out, w_out]
    input_offset = b * (C * H * W) + c * (H * W) + h_out * W + w_out
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    tl.store(output_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def optimized_transpose(k):
    # Transpose: [B, C, H, W] -> [B, C, W, H]
    B, C, H, W = k.shape
    transposed_out = torch.empty((B, C, W, H), device=k.device, dtype=k.dtype)
    
    # Launch transpose kernel
    n_elements = B * C * W * H
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    transpose_kernel[grid](
        k, transposed_out,
        B, C, H, W,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return transposed_out

# Replacement function - must return the function reference
def replacement_func():
    return optimized_transpose