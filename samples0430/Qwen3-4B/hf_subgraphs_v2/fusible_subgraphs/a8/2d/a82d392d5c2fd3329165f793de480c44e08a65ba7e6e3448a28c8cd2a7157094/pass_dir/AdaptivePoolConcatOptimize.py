import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    return torch.cat([torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24)), in_1], dim=1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def adaptive_pool_concat_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_elements)
    
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_0)
    
    adaptive_pool_concat_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
def replacement_func():
    return kernel_wrapper