import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern that matches the exact computation sequence from model.py"""
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    tmp_2 = in_0 == 0
    return tmp_2, tmp_1

def replacement_args(in_0):
    """Extract input tensor for the replacement kernel"""
    return (in_0,)

@triton.jit
def full_computation_kernel(
    input_ptr,
    zero_mask_ptr,
    filled_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    fill_value: tl.constexpr,
):
    """Optimized kernel that performs the full computation sequence"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input value
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute zero mask
    is_zero = (x == 0.0)
    
    # Compute non-zero mask and apply fill
    is_nonzero = ~is_zero
    filled_value = tl.where(is_nonzero, fill_value, x)
    
    # Store results
    tl.store(zero_mask_ptr + offsets, is_zero, mask=mask)
    tl.store(filled_ptr + offsets, filled_value, mask=mask)

@torch.fx.wrap
def full_computation_optimized(in_0):
    """Wrapper function that launches the optimized kernel"""
    n_elements = in_0.numel()
    shape = in_0.shape
    device = in_0.device
    
    # Create output tensors
    zero_mask_out = torch.zeros(shape, dtype=torch.bool, device=device)
    filled_out = torch.empty_like(in_0)
    
    # Kernel configuration with autotuning hints
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    full_computation_kernel[(num_programs,)](
        input_ptr=in_0,
        zero_mask_ptr=zero_mask_out,
        filled_ptr=filled_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        fill_value=-1000.0,
    )
    
    return zero_mask_out, filled_out

def replacement_func():
    """Returns the optimized function"""
    return full_computation_optimized