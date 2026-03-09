import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern matches: interpolate + permute + reshape operations"""
    # Use the dynamic size from the input tensor
    interpolate_size = (y.shape[2], y.shape[3])
    t1 = torch.nn.functional.interpolate(y, size=interpolate_size, mode='bilinear')
    t2 = t1.permute(0, 2, 3, 1)
    spatial_elements = y.shape[2] * y.shape[3]  # H * W
    out1 = t2.reshape(spatial_elements, -1)
    out2 = x[slice(spatial_elements, None, None)]
    return out1, out2

def replacement_args(x, y):
    """Extract arguments for the replacement kernel"""
    return (x, y)


@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return triton_add