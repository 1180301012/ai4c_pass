import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation from model.py
def pattern(in_0, in_1, in_2, in_3):
    in_0 += in_1
    tmp_0 = in_0
    tmp_0 += in_3
    tmp_1 = tmp_0
    tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
    tmp_3 = in_2.chunk(2, dim=1)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_2.chunk(2, dim=1)
    tmp_7 = tmp_6[0]
    tmp_8 = tmp_6[1]
    return (tmp_4, tmp_7, tmp_5, tmp_8)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized Triton kernel for fused add + add + relu
@triton.jit
def fused_add_add_relu_kernel(
    in_0_ptr,
    in_1_ptr, 
    in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    a = tl.load(in_0_ptr + offsets, mask=mask)
    b = tl.load(in_1_ptr + offsets, mask=mask)
    c = tl.load(in_3_ptr + offsets, mask=mask)
    
    # Fused computation: add + add + relu
    result = a + b + c
    result = tl.maximum(result, 0.0)  # relu
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper - decorated with torch.fx.wrap
@torch.fx.wrap
def fused_add_add_relu_chunk_wrapper(in_0, in_1, in_2, in_3):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor for the fused add+add+relu
    out = torch.empty_like(in_0)
    
    # Launch the fused kernel
    fused_add_add_relu_kernel[(num_blocks,)](
        in_0,
        in_1,
        in_3,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Chunk operations (these are just views, no computation needed)
    tmp_3 = in_2.chunk(2, dim=1)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = out.chunk(2, dim=1)
    tmp_7 = tmp_6[0]
    tmp_8 = tmp_6[1]
    
    return (tmp_4, tmp_7, tmp_5, tmp_8)

# Replacement function - returns the kernel wrapper function (not a call)
def replacement_func():
    return fused_add_add_relu_chunk_wrapper