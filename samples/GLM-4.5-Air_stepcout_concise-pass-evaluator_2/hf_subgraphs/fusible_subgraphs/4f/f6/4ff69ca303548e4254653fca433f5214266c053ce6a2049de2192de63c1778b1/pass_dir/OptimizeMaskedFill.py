import torch
import triton
import triton.language as tl

def pattern(input_tensor, mask, fill_value):
    """Pattern that matches the masked_fill operation"""
    result = input_tensor.masked_fill(mask, fill_value)
    return result

def replacement_args(input_tensor, mask, fill_value):
    """Extract arguments for the replacement kernel"""
    return (input_tensor, mask, fill_value)

@triton.jit
def optimized_masked_fill_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for masked fill operation with better memory access"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and mask values with cache hints
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0)
    
    # Apply masked fill - use proper Triton boolean handling
    result = tl.where(mask_val, fill_value, x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(input_tensor, mask, fill_value):
    """Wrapper function that launches the optimized kernel"""
    n_elements = input_tensor.numel()
    shape = input_tensor.shape
    device = input_tensor.device
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Kernel configuration - optimized for tensor size ~870k elements
    BLOCK_SIZE = 8192  # Larger block for better GPU utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_masked_fill_kernel[(num_programs,)](
        input_ptr=input_tensor,
        mask_ptr=mask,
        output_ptr=output,
        n_elements=n_elements,
        fill_value=fill_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns the optimized function"""
    return optimized_masked_fill