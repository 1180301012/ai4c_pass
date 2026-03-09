import torch
import triton
import triton.language as tl

# Pattern matching function: matches one-input addition pattern
def pattern(in_0):
    # tmp_0 = 0 + in_0
    tmp_0 = 0 + in_0
    # tmp_0 += 0
    tmp_0 += 0
    # tmp_1 = tmp_0
    tmp_1 = tmp_0
    # tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    # return (tmp_1, tmp_2)
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for one-input fused addition
@triton.jit
def fused_one_add_kernel(
    x_ptr,
    out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * height * width)
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform one-input operation: x + 0 (which is just x, but for completeness)
    out = x + 0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_one_add_wrapper(in_0):
    batch_size, channels, height, width = in_0.shape
    n_elements = batch_size * channels * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor for fused addition (same shape as input)
    out = torch.empty_like(in_0)
    
    # Launch kernel for fused addition
    fused_one_add_kernel[(num_programs,)](
        in_0,
        out,
        batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # For now, return the addition result
    # The mean operation will be handled by a separate pass
    return out, out.mean((2, 3), keepdim=True)

# Replacement function
def replacement_func():
    return fused_one_add_wrapper