import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0):
    # Full computation pattern from the model:
    # tmp_0 = in_0.to(torch.float32)
    # tmp_1 = torch.tensor(1.0, dtype=torch.float32)
    # tmp_2 = tmp_1 - tmp_0
    # tmp_3 = tmp_2.to(torch.bool)
    # tmp_4 = tmp_2.masked_fill(tmp_3, -3.4028234663852886e+38)
    x = in_0.to(torch.float32)
    one = torch.tensor(1.0, dtype=torch.float32)
    diff = one - x
    mask = diff.to(torch.bool)
    out = diff.masked_fill(mask, -3.4028234663852886e+38)
    return out

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel that fuses all operations
@triton.jit
def fused_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as float32 (implicitly converts from int64)
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute 1.0 - x
    result = 1.0 - x
    
    # Convert to bool (True where result != 0)
    # Then apply: if True, use -inf, else keep result
    # This is equivalent to: result == 0 ? result : -inf
    neg_inf = float("-inf")
    result = tl.where(result != 0.0, neg_inf, result)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0):
    # Get number of elements
    n_elements = in_0.numel()
    
    # Determine block size - use 1024 as default
    BLOCK_SIZE = 1024
    
    # Calculate grid
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    # Launch kernel
    fused_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper