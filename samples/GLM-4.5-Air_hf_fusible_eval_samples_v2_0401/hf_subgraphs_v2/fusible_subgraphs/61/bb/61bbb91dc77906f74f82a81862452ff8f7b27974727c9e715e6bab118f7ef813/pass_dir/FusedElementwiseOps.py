import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_5, in_0, in_2):
    """Match fused element-wise operations: multiply with scalar + addition"""
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    return tmp_4

# Argument extraction function
def replacement_args(in_5, in_0, in_2):
    return (in_5, in_0, in_2)

@triton.jit
def fused_elementwise_kernel(
    in_5_ptr, in_0_value, in_2_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    in_5 = tl.load(in_5_ptr + offsets, mask=mask)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask)
    
    # Apply fused operations: multiply with scalar then add
    out = in_5 * in_0_value + in_2
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_elementwise_ops(in_5, in_0, in_2):
    # Broadcast scalar to tensor if needed
    if in_0.numel() == 1:
        scalar_value = in_0.item()
    else:
        scalar_value = in_0  # Tensor broadcast case
    
    n_elements = in_5.numel()
    out = torch.empty_like(in_5)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_elementwise_kernel[(num_programs,)](
        in_5_ptr=in_5,
        in_0_value=scalar_value,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_elementwise_ops