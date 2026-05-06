import torch
import triton
import triton.language as tl

def pattern(in_0, in_2):
    n = in_2.shape[0]
    range_tensor = torch.arange(n, device=in_2.device)
    positions = in_2.view(-1, 1)
    mask = range_tensor <= positions
    mask_expanded = mask.expand(1, 1, n, n)
    return mask_expanded

def replacement_args(in_0, in_2):
    return (in_0, in_2)

@triton.jit
def mascara_kernel(
    positions_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= n:
        return
    
    for j in range(BLOCK_SIZE):
        pos = tl.load(positions_ptr + j)
        mask = i <= pos
        tl.store(out_ptr + (i * n) + j, mask)

@torch.fx.wrap
def mascara_kernel_wrapper(positions):
    n = positions.shape[0]
    out = torch.zeros((n, n), dtype=torch.bool, device=positions.device)
    mascara_kernel[(tl.cdiv(n, 64),)](
        positions_ptr=positions,
        out_ptr=out,
        n=n,
        BLOCK_SIZE=64
    )
    return out

def replacement_func():
    return mascara_kernel_wrapper