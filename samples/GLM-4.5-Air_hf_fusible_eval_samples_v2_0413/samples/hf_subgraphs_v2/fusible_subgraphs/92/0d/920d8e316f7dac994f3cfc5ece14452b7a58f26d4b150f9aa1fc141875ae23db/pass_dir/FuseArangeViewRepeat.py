import torch
from torch import device
import triton
import triton.language as tl

def pattern():
    tmp_0 = torch.arange(0, 128, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return (tmp_2,)

def replacement_args():
    return ()

@triton.jit
def fused_arange_repeat_kernel(out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    local_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + local_offsets
    mask = offsets < N
    
    # Generate values directly: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    values = offsets
    
    # Store in row 0 (offset = j)
    tl.store(out_ptr + offsets, values, mask=mask)
    # Store in row 1 (offset = N + j)
    tl.store(out_ptr + N + offsets, values, mask=mask)

@torch.fx.wrap  
def fused_arange_repeat():
    N = 128
    out = torch.empty((2, N), dtype=torch.int32, device='cuda')
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_arange_repeat_kernel[(num_programs,)](
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return fused_arange_repeat