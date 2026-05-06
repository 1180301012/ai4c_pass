import torch
import triton
import triton.language as tl

def pattern(x):
    relu_out = torch.nn.functional.relu(x, inplace=True)
    return relu_out.flatten(1, -1)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_relu_flatten_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty(n_elements, dtype=x.dtype, device=x.device)
    optimized_relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(x.shape[0], x.shape[1])

def replacement_func():
    return kernel_wrapper