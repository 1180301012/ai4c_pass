import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the conversion and dropout sequence
    x_float = x.float()
    # Simulate softmax operation with type conversion back
    x_orig = x_float.type_as(x)
    # Dropout with training=False is just scaling by 0.9
    out = x_orig * 0.9
    return out

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_conversion_dropout_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for better precision, then back to original dtype directly
    # Instead of doing separate conversions, we'll let the wrapper handle it more efficiently
    x_float = x.to(tl.float32)
    
    # Apply dropout scaling (multiply by 0.9 for p=0.1 training=False)
    out = x_float * 0.9
    
    # Store result back as original dtype
    # The dtype will be handled by the wrapper
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_conversion_dropout(x):
    n_elements = x.numel()
    device = x.device
    orig_dtype = x.dtype
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32)  # Use float32 for precision
    
    # Block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_conversion_dropout_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert back to original dtype
    return out.to(orig_dtype)

def replacement_func():
    return optimized_conversion_dropout