import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, y):
    tmp_0 = y + x  # Note: order is y + x to match original in_1 + in_0
    tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
    return (tmp_1,)

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized kernel for addition + SiLU fusion
@triton.jit
def add_silu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Elementwise addition + SiLU activation in one step
    add_result = x + y
    silu_result = add_result * tl.sigmoid(add_result)
    
    # Store result
    tl.store(out_ptr + offsets, silu_result, mask=mask)

@torch.fx.wrap
def fused_add_silu(x, y):
    # Ensure inputs have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Choose appropriate block size for good GPU utilization
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    add_silu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_add_silu