import torch
import triton
import triton.language as tl

# Simple test pattern - same as scalar multiplication to test loading
def pattern(in_1):
    """Test pattern to check if loading works"""
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    """Extract input tensor"""
    return (in_1,)

# Simple Triton kernel for testing
@triton.jit
def test_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple test kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = x * scalar_val
    tl.store(output_ptr + offsets, result, mask=mask)

# Test kernel wrapper
@torch.fx.wrap
def test_function(x):
    """Simple test function"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    test_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        scalar_val=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Replacement function
def replacement_func():
    return test_function