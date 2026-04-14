import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation in model.py
def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel with memory optimizations and fewer instructions"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Combined operations: convert, subtract, and mask in fewer steps
    in_float = tl.cast(tl.load(in_ptr + offsets, mask=mask, other=0), tl.float32)
    one_minus_x = 1.0 - in_float
    
    # Optimized: use the neg_inf value directly without unnecessary memory operations
    result = tl.where(one_minus_x <= 0.0, -3.4028234663852886e+38 * one_minus_x, one_minus_x * one_minus_x)
    
    # Store result using direct store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def compute_optimized(x):
    """Optimized kernel wrapper with balanced performance"""
    n_elements = x.numel()
    
    # Use balanced optimization for small tensors
    BLOCK_SIZE = 128  # Good trade-off between granularity and overhead
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape and dtype as original result  
    output = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    
    # Launch optimized kernel with block size that balances efficiency
    optimized_kernel[(num_programs,)](
        in_ptr=x,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return compute_optimized