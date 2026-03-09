import torch
import triton
import triton.language as tl

# Pattern matching function: matches three-input addition pattern
def pattern(in_0, in_1, in_2):
    # tmp_0 = in_1 + in_2
    tmp_0 = in_1 + in_2
    # tmp_0 += in_0
    tmp_0 += in_0
    # tmp_1 = tmp_0
    tmp_1 = tmp_0
    # tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    # return (tmp_1, tmp_2)
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel for three-input fused addition
@triton.jit
def fused_three_add_kernel(
    x_ptr, y_ptr, z_ptr,
    out_ptr,
    mean_out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * height * width)
    
    # Load all three inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Perform fused three-input addition: x + y + z
    out = x + y + z
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_three_add_wrapper(in_0, in_1, in_2):
    batch_size, channels, height, width = in_0.shape
    n_elements = batch_size * channels * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor for fused addition (same shape as input)
    out = torch.empty_like(in_0)
    
    # Launch kernel for fused addition
    fused_three_add_kernel[(num_programs,)](
        in_0, in_1, in_2,
        out,
        None,  # mean_out_ptr not used in this pass
        batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # For now, return the addition result
    # The mean operation will be handled by a separate pass
    return out, out.mean((2, 3), keepdim=True)

# Replacement function
def replacement_func():
    return fused_three_add_wrapper