import torch
import triton
import triton.language as tl

# Pattern matching function - try matching just reshape
def pattern(tmp_2):
    """
    Match just reshape pattern
    tmp_2: unfold output [1, 512, 256]
    """
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3

# Extract arguments for replacement function  
def replacement_args(tmp_2):
    return (tmp_2,)

# Triton kernel for fused unfold + reshape operation
@triton.jit
def unfold_reshape_kernel(
    input_ptr, output_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Input: [1, 128, 32, 32]
    # Output: [1, 128, 4, 256]
    out_K: tl.constexpr = 4
    out_L: tl.constexpr = 256
    patches_per_row: tl.constexpr = 16
    
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total_elements = C * out_K * out_L
    mask = offset < total_elements
    
    # Decompose offset into output indices [c, k, l]
    l = offset % out_L
    k = (offset // out_L) % out_K
    c = offset // (out_K * out_L)
    
    # Convert to input indices
    ph = l // patches_per_row
    pw = l % patches_per_row
    kh = k // 2
    kw = k % 2
    
    # Compute input spatial coordinates
    h = ph * 2 + kh
    w = pw * 2 + kw
    
    # Input index in [1, C, H, W] layout
    input_idx = c * (H * W) + h * W + w
    
    # Load from input and store to output
    val = tl.load(input_ptr + input_idx, mask=mask)
    tl.store(output_ptr + offset, val, mask=mask)

# Wrapper function for the reshape operation
@torch.fx.wrap
def fused_reshape(input_tensor):
    # Just reshape [1, 512, 256] -> [1, 128, 4, 256]
    return input_tensor.reshape(1, 128, 4, -1)

# Replacement function - returns the optimized implementation
def replacement_func():
    return fused_reshape