import torch
import triton
import triton.language as tl

def pattern(x):
    # Hardtanh: clip values to range [0.0, 6.0]
    result = torch.nn.functional.hardtanh(x, 0.0, 6.0, False)
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply hardtanh: clip between 0.0 and 6.0
    # Using efficient minimum + maximum operations
    result = tl.maximum(x, 0.0)
    result = tl.minimum(result, 6.0)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_hardtanh(x):
    # Get total number of elements
    n_elements = x.numel()
    
    # Dynamically choose optimal block size based on tensor size
    if n_elements < 100000:
        BLOCK_SIZE = 1024   # Small tensors
    elif n_elements < 1000000:
        BLOCK_SIZE = 2048   # Medium tensors  
    elif n_elements < 10000000:
        BLOCK_SIZE = 4096   # Large tensors
    else:
        BLOCK_SIZE = 8192   # Very large tensors
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x, device=x.device)
    
    # Launch optimized kernel
    hardtanh_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_hardtanh