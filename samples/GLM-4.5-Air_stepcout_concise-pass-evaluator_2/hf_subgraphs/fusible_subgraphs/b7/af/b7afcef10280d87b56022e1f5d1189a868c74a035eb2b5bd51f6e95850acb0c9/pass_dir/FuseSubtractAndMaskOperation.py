import torch
import triton
import triton.language as tl

def pattern(x):
    return x.to(torch.float32)

def replacement_args(x):
    return (x,)

@triton.jit
def type_conversion_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data (int64)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32
    result = input_val.to(tl.float32)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def type_conversion_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data (int64)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32
    result = input_val.to(tl.float32)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def type_conversion_operation(x):
    # Get input tensor properties
    input_shape = x.shape
    input_dtype = x.dtype
    device = x.device
    
    # Ensure input is on the same device
    if x.device.type != 'cuda':
        x = x.cuda()
    
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with float32 dtype
    out = torch.empty(input_shape, dtype=torch.float32, device=device)
    
    # Launch Triton kernel with autotuning
    type_conversion_kernel[(num_programs,)](
        input_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return type_conversion_operation