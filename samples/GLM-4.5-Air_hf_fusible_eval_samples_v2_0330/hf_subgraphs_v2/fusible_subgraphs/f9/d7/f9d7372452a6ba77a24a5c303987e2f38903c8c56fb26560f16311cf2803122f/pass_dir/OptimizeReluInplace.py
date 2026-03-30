import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    # Match the ReLU operation: tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def optimized_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU operation - optimized for float16/bfloat16
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu(x):
    # Reshape inputs to 1D for contiguous memory access
    x_flat = x.view(-1)
    N = x_flat.numel()
    
    # Launch kernel with optimal block size
    BLOCK_SIZE = 512
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x_flat)
    
    optimized_relu_kernel[(num_programs,)](
        input_ptr=x_flat,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return out.view(x.shape)

def replacement_func():
    return optimized_relu