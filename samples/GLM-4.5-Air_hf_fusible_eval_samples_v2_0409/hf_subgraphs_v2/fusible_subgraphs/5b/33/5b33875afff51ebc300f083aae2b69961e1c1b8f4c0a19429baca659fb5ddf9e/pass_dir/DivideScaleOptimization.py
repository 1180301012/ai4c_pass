import torch
import triton
import triton.language as tl

@triton.jit
def divide_kernel(
    x_ptr, scale_ptr, out_ptr,
    total_elements, BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input and scale
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division
    out = x / scale
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_divide(input_tensor, scale):
    # Handle the case where scale is a constant
    if isinstance(scale, (int, float)):
        # For constant scale, we can still benefit from optimized kernel
        return input_tensor / scale
    else:
        # For tensor scale, use optimized triton kernel with autotuning
        total_elements = input_tensor.numel()
        
        # Choose optimal block size based on tensor size
        if total_elements < 1024:
            BLOCK_SIZE = 64
        elif total_elements < 10000:
            BLOCK_SIZE = 256
        elif total_elements < 100000:
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 1024
        
        total_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty_like(input_tensor)
        
        # Use basic kernel launch without explicit warp specification
        divide_kernel[total_programs](
            input_tensor, scale, output,
            total_elements, BLOCK_SIZE
        )
        
        return output

def pattern(x, scale):
    # Simple division pattern
    return x / scale

def replacement_args(x, scale):
    return (x, scale)

def replacement_func():
    return optimized_divide