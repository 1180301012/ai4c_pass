import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    """Simple test pattern"""
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    return tmp_6

def replacement_args(tmp_1):
    return (tmp_1,)

@triton.jit
def simple_cos_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple cosine kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data as bf16 and convert to fp32 for computation
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute cosine and apply identity multiplication
    result = tl.cos(x)
    
    # Convert back to bfloat16 for output
    result_bf16 = result.to(tl.bfloat16)
    
    # Store results
    tl.store(output_ptr + offsets, result_bf16, mask=mask)

@torch.fx.wrap
def simple_cos_func(tmp_1):
    """Simple cosine function wrapper"""
    # Get input size
    input_size = tmp_1.numel()
    
    # Create output tensor
    output = torch.empty_like(tmp_1, dtype=torch.bfloat16)
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    simple_cos_kernel[(num_programs,)](
        input_ptr=tmp_1,
        output_ptr=output,
        n_elements=input_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return simple_cos_func