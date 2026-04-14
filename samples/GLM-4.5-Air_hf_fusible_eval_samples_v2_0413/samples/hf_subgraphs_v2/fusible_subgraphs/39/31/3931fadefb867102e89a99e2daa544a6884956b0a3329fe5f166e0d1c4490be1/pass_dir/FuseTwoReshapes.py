import torch
import triton
import triton.language as tl

def pattern(x):
    # Match two consecutive reshape operations
    tmp_1 = x.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def fused_reshape_kernel(
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
    
    # Store output data (reshape is just a view change, no computation needed)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_reshape(x):
    # Input shape: [1, 124, 1536]
    # Output shape: [1, 248, 768]
    N = x.numel()
    BLOCK_SIZE = 2048
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(1, 248, 768, dtype=x.dtype, device=x.device)
    
    fused_reshape_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_reshape