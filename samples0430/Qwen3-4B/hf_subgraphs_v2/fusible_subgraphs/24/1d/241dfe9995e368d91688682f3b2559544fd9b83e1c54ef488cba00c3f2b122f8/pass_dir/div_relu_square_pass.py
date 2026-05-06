import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp0 = in_0 / 11.313708498984761
    tmp1 = torch.nn.functional.relu(tmp0)
    tmp2 = torch.square(tmp1)
    return (tmp2,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block index
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x / SCALE
    x = tl.where(x > 0, x, 0.0)
    out = x * x
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    div_relu_square_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        SCALE=11.313708498984761,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return kernel_wrapper