import torch
import triton
import triton.language as tl

# Pattern matching function: matches three-input addition pattern
def pattern_three_in(in_0, in_1, in_2):
    # tmp_0 = in_1 + in_2
    tmp_0 = in_1 + in_2
    # tmp_0 += in_0  
    tmp_0 += in_0
    # tmp_1 = tmp_0
    tmp_1 = tmp_0
    # tmp_0 = None (excluded from pattern as it's cleanup)
    # tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    # return (tmp_1, tmp_2)
    return (tmp_1, tmp_2)

# Pattern matching function: matches two-input addition pattern (with zero)
def pattern_two_in(in_0, in_1):
    # tmp_0 = 0 + in_1
    tmp_0 = 0 + in_1
    # tmp_0 += in_0
    tmp_0 += in_0
    # tmp_1 = tmp_0
    tmp_1 = tmp_0
    # tmp_0 = None (excluded from pattern as it's cleanup)
    # tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    # return (tmp_1, tmp_2)
    return (tmp_1, tmp_2)

# Pattern matching function: matches one-input addition pattern
def pattern_one_in(in_0):
    # tmp_0 = 0 + in_0
    tmp_0 = 0 + in_0
    # tmp_0 += 0
    tmp_0 += 0
    # tmp_1 = tmp_0
    tmp_1 = tmp_0
    # tmp_0 = None (excluded from pattern as it's cleanup)
    # tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    # return (tmp_1, tmp_2)
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2=None):
    if in_2 is not None:
        return (in_0, in_1, in_2)
    else:
        return (in_0, in_1)

# Optimized Triton kernel for fused addition
@triton.jit
def fused_add_kernel(
    x_ptr, y_ptr, z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # If third input exists, load it too
    if z_ptr is not None:
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        out = x + y + z
    else:
        out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap  
def fused_add_wrapper(in_0, in_1, in_2=None):
    # Determine input shape
    if in_2 is not None:
        # Three-input case
        n_elements = in_0.numel()
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = torch.empty_like(in_0)
        
        fused_add_kernel[(num_programs,)](
            in_0, in_1, in_2,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Two-input case (treat in_2 as None)
        n_elements = in_0.numel()
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = torch.empty_like(in_0)
        
        fused_add_kernel[(num_programs,)](
            in_0, in_1, None,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function (returns the optimized kernel)
def replacement_func():
    return fused_add_wrapper