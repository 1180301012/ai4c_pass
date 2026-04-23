import torch
import triton
import triton.language as tl

def pattern(x, p, train, inplace):
    return torch.nn.functional.dropout(x, p, train, inplace)

def replacement_args(x, p, train, inplace):
    return (x, p, train, inplace)

@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    p,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Better PRNG using thread_id for deterministic random
    seed = offsets * 314159265 + 271828182
    rand = tl.cast(seed, tl.uint32) % 1000000007
    rand = rand / 1000000007.0  # Convert to [0,1)

    scale = 1.0 / (1.0 - p)
    out = tl.where(rand > p, x * scale, tl.zeros_like(x))

    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def dropout_wrapper(x, p, train, inplace):
    n_elements = x.numel()
    BLOCK_SIZE = 512
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)
    dropout_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        p=p,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return dropout_wrapper