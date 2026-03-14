import torch
import triton
import triton.language as tl

# Pattern matching function - match relu operation
def pattern(x):
    """
    Match relu operation
    
    From model.py:
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    """
    tmp_8 = torch.nn.functional.relu(x, True)
    return tmp_8

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel for relu
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply relu
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


# Wrapper function for the Triton kernel
@torch.fx.wrap
def triton_relu(x):
    n_elements = x.numel()
    
    # Ensure input is contiguous
    x_contig = x.contiguous()
    
    # Output same shape as input
    out = torch.empty_like(x_contig)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    relu_kernel[grid](
        x_ptr=x_contig,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# Replacement function
def replacement_func():
    return triton_relu