import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern matches: ReLU operation alone
    """
    return torch.nn.functional.relu(x, inplace=False)

def replacement_args(x):
    return (x,)

@triton.jit
def triton_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU: max(x, 0)
    relu_x = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_x, mask=mask)

@torch.fx.wrap
def triton_relu(x):
    # Get input tensor shape and total elements
    if x.is_contiguous():
        input_flat = x.flatten()
    else:
        input_flat = x.contiguous().flatten()
    
    n_elements = input_flat.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as input
    out = torch.empty_like(input_flat)
    
    # Launch Triton kernel
    triton_relu_kernel[(num_programs,)](
        x_ptr=input_flat,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match input original shape
    return out.reshape(x.shape)

def replacement_func():
    return triton_relu