import torch
import triton
import triton.language as tl

# Pattern: Just mean operation without keepdim
def pattern(x):
    return x.mean([2, 3])

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['HW'],
)
@triton.jit
def mean_2d_kernel(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    nc_id = tl.program_id(0)
    base_offset = nc_id * HW
    
    acc = 0.0
    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    
    mean_val = acc / HW
    tl.store(output_ptr + nc_id, mean_val)

@torch.fx.wrap
def mean_2d(x):
    N, C, H, W = x.shape
    HW = H * W
    
    output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    num_programs = N * C
    
    mean_2d_kernel[(num_programs,)](
        x,
        output.view(-1),
        HW,
    )
    
    return output

def replacement_func():
    return mean_2d