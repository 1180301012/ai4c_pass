import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the addition operation
    return in_1 + in_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Wrapper function for optimized addition
@torch.fx.wrap
def triton_add(x, y):
    # Optimized Triton addition kernel
    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles a contiguous block of data
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data with masking
        x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        
        # Perform addition
        out = x_val + y_val
        
        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)
    
    # Ensure tensors have the same shape
    assert x.shape == y.shape, f"Tensor shapes must match: {x.shape} vs {y.shape}"
    
    # Get tensor size
    n_elements = x.numel()
    
    # Choose block size based on tensor size
    if n_elements <= 1024:
        BLOCK_SIZE = 256
    elif n_elements <= 10240:
        BLOCK_SIZE = 512
    elif n_elements <= 102400:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Calculate number of programs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add