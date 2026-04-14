import torch
import triton
import triton.language as tl

@triton.jit
def optimized_range_kernel(out_ptr, n_elements, start, BLOCK_SIZE: tl.constexpr):
    """Optimized kernel to create range tensor on GPU"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Create range values: start + offsets
    values = start + offsets.to(tl.float32)  # Use float32 for precision
    
    # Store results
    tl.store(out_ptr + offsets, values, mask=mask)

@triton.jit
def optimized_bool_conversion_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized kernel for int64 to boolean conversion"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    
    # Convert to boolean (non-zero becomes True, zero becomes False)
    bool_vals = input_vals != 0
    
    # Store results
    tl.store(out_ptr + offsets, bool_vals, mask=mask)

@torch.fx.wrap
def optimized_range_and_conversion(in_tensor):
    """Wrapper function that creates both optimized tensors"""
    # Handle range tensor
    range_size = in_tensor.shape[-1]  # Use last dimension size for range
    start = 0
    
    out_range = torch.empty((range_size,), dtype=torch.float32, device=in_tensor.device)
    n_elements = range_size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_range_kernel[(num_programs,)](
        out_range,
        n_elements,
        start,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Handle boolean conversion
    out_bool = torch.empty_like(in_tensor, dtype=torch.bool)
    n_elements_bool = in_tensor.numel()
    num_programs_bool = (n_elements_bool + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_bool_conversion_kernel[(num_programs_bool,)](
        in_tensor,
        out_bool,
        n_elements_bool,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (out_range, out_bool)

def pattern(in_0):
    """Pattern matching the computation: arange + type conversion"""
    from torch import device
    tmp_1 = torch.arange(0, in_0.shape[-1], device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    """Extract arguments for replacement"""
    return (in_0,)

def replacement_func():
    """Return the optimized kernel wrapper"""
    return optimized_range_and_conversion