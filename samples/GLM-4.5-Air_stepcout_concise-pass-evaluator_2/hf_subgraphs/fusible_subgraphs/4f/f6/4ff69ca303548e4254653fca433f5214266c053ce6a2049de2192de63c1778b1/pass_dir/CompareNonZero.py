import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Test pattern for != 0 operation"""
    tmp_0 = in_0 != 0
    return tmp_0

def replacement_args(in_0):
    """Extract input tensor for the replacement kernel"""
    return (in_0,)

@triton.jit
def compare_non_zero_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that compares input to non-zero"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input value
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compare to non-zero
    is_nonzero = (x != 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, is_nonzero, mask=mask)

@torch.fx.wrap
def compare_non_zero(in_0):
    """Wrapper function that launches the optimized kernel"""
    n_elements = in_0.numel()
    shape = in_0.shape
    device = in_0.device
    
    # Create output tensor
    output = torch.zeros(shape, dtype=torch.bool, device=device)
    
    # Kernel configuration
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    compare_non_zero_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns the optimized function"""
    return compare_non_zero