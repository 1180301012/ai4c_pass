import torch
import triton
import triton.language as tl

def pattern(input):
    hardswish_out = torch.nn.functional.hardswish(input, True)
    return hardswish_out

def replacement_args(input):
    return (input,)

@triton.jit
def hardswish_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply hardswish: x * relu6(x + 3) / 6
    # First compute relu6(x + 3)
    x_plus_3 = x + 3.0
    relu6_val = tl.maximum(0.0, tl.minimum(x_plus_3, 6.0))
    
    # Then compute final hardswish result
    hardswish_result = x_plus_3 * relu6_val / 6.0
    
    # Store result
    tl.store(output_ptr + offsets, hardswish_result, mask=mask)

@torch.fx.wrap
def optimized_hardswish(input):
    n_elements = input.numel()
    
    # Configure block size for optimal performance
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    hardswish_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_hardswish