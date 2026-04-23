import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(dst_ptr, src0_ptr, src1_ptr, n0, n1, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n0 + n1) * m
    row = offsets // m
    col = offsets % m
    # Copy from appropriate source
    src0_offset = row * m + col
    src1_offset = (row - n1) * m + col
    val = tl.load(src0_ptr + src0_offset, mask=(row < n1) & mask)
    val = tl.load(src1_ptr + src1_offset, mask=(row >= n1) & mask)
    tl.store(dst_ptr + offsets, val, mask=mask)

def optimized_cat(in_0, in_1):
    n0, n1 = in_0.shape[0], in_1.shape[0]
    m = in_0.shape[1]
    out = torch.empty([n0 + n1, m], dtype=in_0.dtype, device=in_0.device)
    n_elements = (n0 + n1) * m
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    copy_kernel[(num_programs,)](
        dst_ptr=out, src0_ptr=in_1, src1_ptr=in_0,
        n0=n0, n1=n1, m=m, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(in_0, in_1):
    return torch.cat([in_1, in_0])

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return optimized_cat