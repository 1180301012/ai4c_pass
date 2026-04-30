import torch
import triton
import triton.language as tl

@triton.jit
def flatten_kernel(out_ptr, in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def flatten_wrapper(tmp_22):
    n_elements = tmp_22.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if tmp_22.device.type == 'cpu':
        tmp_22 = tmp_22.cuda()
    
    out = torch.empty((n_elements,), dtype=tmp_22.dtype, device='cuda')
    
    flatten_kernel[(num_programs,)](
        out_ptr=out,
        in_ptr=tmp_22,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(tmp_22):
    tmp_28 = tmp_22.view(-1)
    return tmp_28

def replacement_args(tmp_22):
    return (tmp_22,)

def replacement_func():
    return flatten_wrapper