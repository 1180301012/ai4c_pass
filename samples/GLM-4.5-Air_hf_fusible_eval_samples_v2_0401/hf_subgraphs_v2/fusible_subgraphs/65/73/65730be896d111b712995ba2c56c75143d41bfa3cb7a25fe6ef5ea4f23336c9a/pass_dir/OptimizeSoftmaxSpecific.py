import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the softmax computation with float conversion
    x_float = x.float()
    softmax_out = torch.nn.functional.softmax(x_float, dim=-1)
    return softmax_out.type_as(x)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_softmax_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input (convert to float32 for numerical stability)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Find max for stable softmax
    max_val = tl.max(x, mask=mask)
    
    # Compute exp(z - max)
    exp_x = tl.exp(x - max_val)
    
    # Sum for normalization within the block
    sum_exp = tl.sum(exp_x, mask=mask)
    
    # Softmax
    softmax_out = exp_x / sum_exp
    
    # Store result as original dtype
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def optimized_softmax(x):
    n_elements = x.numel()
    device = x.device
    orig_dtype = x.dtype
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_softmax_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_softmax