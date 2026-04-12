import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    """
    Pattern matches the sequence of operations:
    1. Convert int64 to float32
    2. Compute 1.0 - x  
    3. Convert to boolean and masked_fill
    4. Multiply by original
    """
    tmp_0 = x.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel using Triton
@triton.jit
def fused_comp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as float32 directly (convert from int64 on the fly)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Fuse the computation: (1.0 - x) * ((1.0 - x) if (1.0 - x) != 0 else -3.4028234663852886e+38)
    # But since we're doing boolean conversion, it's equivalent to:
    # Compute y = 1.0 - x
    # Convert y to boolean (y != 0)
    # masked_fill with large negative number for non-zero y values? Actually wait, let me think about this...
    # In PyTorch, tmp_2 = tmp_1.bool() converts to boolean: 0.0 -> False, anything non-zero -> True
    # tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38) replaces True values with the large negative number
    
    # So the computation is: 
    # y = 1.0 - x
    # result = y * (y if y == 0 else -3.4028234663852886e+38)
    
    # But let's look at this more carefully:
    # tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    # tmp_2 = tmp_1.bool(), so:
    # If tmp_1 is 0.0 (which is False), then tmp_3 = tmp_1 = 0.0
    # If tmp_1 is non-zero (which is True), then tmp_3 = -3.4028234663852886e+38
    
    # Then tmp_4 = tmp_3 * tmp_1
    
    # So: result = (tmp_1 if tmp_1 == 0 else -3.4028234663852886e+38) * tmp_1
    # Let's simplify:
    # If tmp_1 = 0: result = 0 * 0 = 0
    # If tmp_1 != 0: result = -3.4028234663852886e+38 * tmp_1
    
    # This is a very specific mathematical operation that might be optimized directly in Triton
    
    # Direct calculation with conditional logic
    y = 1.0 - input_vals
    zero_mask = y == 0.0
    
    # For values where y == 0, result = y * y = 0
    # For values where y != 0, result = (-3.4028234663852886e+38) * y
    result = tl.where(zero_mask, y * y, -3.4028234663852886e+38 * y)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_computation(in_0):
    """Fused computation that performs the entire operation sequence in one kernel"""
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0, dtype=torch.float32)

    fused_comp_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_computation