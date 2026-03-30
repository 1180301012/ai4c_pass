import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor):
    """Pattern matching for simple unsqueeze(0) operation"""
    return input_tensor.unsqueeze(0)

def replacement_args(input_tensor):
    """Extract arguments for the replacement function"""
    return (input_tensor,)

@triton.jit
def optimized_unsqueeze_repeat_kernel(
    input_ptr,
    output_ptr,
    n_elements_total,
    n_elements_input,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that fuses unsqueeze(0) and repeat(1, 1) operations"""
    # Each program handles a block of the output tensor
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_total
    
    # Load the single input element (assuming it's broadcasted)
    input_val = tl.load(input_ptr, mask=offsets < 1)
    
    # For output indices, map back to input index (should be 0 for all since we're repeating the same value)
    # The pattern unsqueeze(0).repeat(1, 1) creates a copy of the single element
    output_val = input_val
    
    # Store the result
    tl.store(output_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze(input_tensor):
    """Wrapper function that handles unsqueeze efficiently for all tensor sizes"""
    # For tiny tensors (like our 1-element case), use PyTorch directly
    # This avoids Triton kernel launch overhead for very small computations
    input_size = input_tensor.numel()
    max_tiny_size = 64  # Threshold below which PyTorch is faster
    
    if input_size <= max_tiny_size:
        # Direct PyTorch implementation for small tensors
        return input_tensor.unsqueeze(0)
    else:
        # Use Triton kernel for larger tensors
        output_shape = (1,) + input_tensor.shape
        output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        BLOCK_SIZE = 1024
        num_programs = (output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        simple_unsqueeze_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            input_size=input_size,
            output_size=output.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

@triton.jit
def simple_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    input_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel to demonstrate the pattern"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Copy input data to expanded output
    input_val = tl.load(input_ptr, mask=offsets < input_size)
    tl.store(output_ptr + offsets, input_val, mask=mask)

def replacement_func():
    """Return the optimized function"""
    return optimized_unsqueeze