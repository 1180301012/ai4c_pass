import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """Match multiply + add + ReLU pattern with in-place operations"""
    # This pattern assumes the first pass has already been applied, so we start with:
    tmp_3 = in_1  # This would be the result of fused_sigmoid_view_expand_multiply from pass 1
    tmp_3 += in_0  # In-place addition
    tmp_4 = tmp_3  # Assignment 
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)  # In-place ReLU
    return tmp_5  # Return the ReLU result

# Argument extraction function  
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel
@triton.jit
def fused_add_relu_inplace_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel that performs: (in_1 + in_0) then applies ReLU"""
    pid = tl.program_id(0)
    
    # Calculate total elements and grid
    total_elements = N * C * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load both tensors
    val_in_1 = tl.load(in_1_ptr + offsets, mask=mask)
    val_in_0 = tl.load(in_0_ptr + offsets, mask=mask)
    
    # Perform addition and ReLU in one operation
    # This replaces: tmp_3 += in_0 -> tmp_5 = relu(tmp_4) 
    # Where tmp_4 = tmp_3 and tmp_3 = in_1 + in_0
    result = tl.maximum(val_in_1 + val_in_0, 0.0)
    
    # Store result directly (simulating in-place behavior)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_add_relu_inplace(in_0, in_1, in_2):
    N, C, H, W = in_1.shape
    
    # Set up block size and grid
    BLOCK_SIZE = 1024
    total_elements = N * C * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch kernel
    fused_add_relu_inplace_kernel[(num_programs,)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_add_relu_inplace