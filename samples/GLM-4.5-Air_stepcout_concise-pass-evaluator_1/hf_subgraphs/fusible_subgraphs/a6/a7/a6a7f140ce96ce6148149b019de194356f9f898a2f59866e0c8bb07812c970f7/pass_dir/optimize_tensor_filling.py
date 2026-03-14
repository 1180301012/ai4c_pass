import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern: Tensor slicing, filling with 1s, and cleanup operations
    Replaces multiple separate operations with a single optimized kernel
    """
    tmp_1 = input_tensor[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1)
    tmp_3 = input_tensor[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
    tmp_4 = tmp_3.fill_(1)
    return tmp_2, tmp_4

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_tensor_filling_kernel(
    input_ptr,
    output1_ptr,
    output2_ptr,
    shape,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate total number of elements
    total_elements = shape[0] * shape[1] * shape[2]
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data  
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Create output tensors with zeros
    output1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    output2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # For each offset, determine if it should be filled with 1 based on slicing patterns
    for i, offset in enumerate(offsets):
        if mask[i]:
            # Convert linear offset to 3D coordinates
            batch = offset // (shape[1] * shape[2])
            remainder = offset % (shape[1] * shape[2])
            row = remainder // shape[2] 
            col = remainder % shape[2]
            
            # Pattern 1: slice(-5, None, None) on first dimension, all on third 
            # This means: last 5 rows, all columns
            if batch < shape[0] and (row >= shape[1] - 5) and col < shape[2]:
                output1[i] = 1.0
            
            # Pattern 2: all on first dimension, slice(-5, None, None) on third dimension
            # This means: all rows, last 5 columns
            if batch < shape[0] and row < shape[1] and (col >= shape[2] - 5):
                output2[i] = 1.0
    
    # Store results
    tl.store(output1_ptr + offsets, output1, mask=mask)
    tl.store(output2_ptr + offsets, output2, mask=mask)

@torch.fx.wrap
def optimized_tensor_filling(input_tensor):
    n_elements = input_tensor.numel()
    shape = input_tensor.shape
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output1 = torch.empty_like(input_tensor)
    output2 = torch.empty_like(input_tensor)
    
    optimized_tensor_filling_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output1_ptr=output1,
        output2_ptr=output2,
        shape=shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply slicing operations after filling
    result1 = output1[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    result2 = output2[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
    
    return result1, result2

def replacement_func():
    return optimized_tensor_filling