import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern matching: arange + multiply + reshape + broadcast + flatten
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.arange(0, in_2)
    tmp_3 = tmp_2 * tmp_1
    tmp_4 = tmp_3.view((1,))
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 + tmp_0
    tmp_7 = tmp_6.view(-1)
    return tmp_7

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_flatten_kernel(
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
    
    # Load input data directly from original tensor
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Store directly to output (no computation needed - just data movement)
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_flatten_gpu(in_0):
    """
    Optimized version that eliminates intermediate operations:
    - No need to create arange on GPU
    - No need for scalar multiplication  
    - No need for multiple reshaping operations
    - Direct flattening with efficient memory access
    """
    # Move input to GPU if it's not already there
    if in_0.device != 'cuda':
        in_0 = in_0.to('cuda')
    
    # Get flattened shape
    output_shape = (in_0.numel(),)
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Direct copy from input to flattened output
    optimized_flatten_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_flatten_gpu