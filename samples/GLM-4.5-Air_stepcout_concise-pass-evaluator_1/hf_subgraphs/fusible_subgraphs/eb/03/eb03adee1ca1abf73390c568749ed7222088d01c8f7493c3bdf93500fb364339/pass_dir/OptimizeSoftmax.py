import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    tmp_3 = tmp_2.softmax(dim=-1)
    return tmp_3

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def optimized_softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input block
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Compute exponential
    exp_x = tl.exp(x - max_val)
    
    # Compute sum and normalize
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp
    
    # Store result
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def optimized_softmax(x):
    # Flatten the tensor while keeping track of original shape
    original_shape = x.shape
    N = x.numel()
    
    # Use appropriate block size for softmax
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    optimized_softmax_kernel[(num_programs,)](
        x,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_softmax