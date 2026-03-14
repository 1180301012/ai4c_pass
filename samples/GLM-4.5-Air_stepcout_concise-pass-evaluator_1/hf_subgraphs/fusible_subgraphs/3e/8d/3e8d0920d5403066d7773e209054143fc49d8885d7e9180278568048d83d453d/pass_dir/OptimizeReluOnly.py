import torch
import triton
import triton.language as tl


def pattern(input_in):
    tmp_9 = torch.nn.functional.relu(input_in, inplace=False)
    return tmp_9


def replacement_args(input_in):
    return (input_in,)


@triton.jit
def optimized_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_relu(input_in):
    n_elements = input_in.numel()
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_in)
    
    # Launch kernel
    optimized_relu_kernel[(num_programs,)](
        input_ptr=input_in,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return optimized_relu