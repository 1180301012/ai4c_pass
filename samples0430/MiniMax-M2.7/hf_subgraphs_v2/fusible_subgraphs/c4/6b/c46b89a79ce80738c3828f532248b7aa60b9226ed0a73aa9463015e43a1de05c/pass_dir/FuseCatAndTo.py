import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(out_ptr, in_ptr, offset, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load as float32 first for proper precision, then convert
    val = tl.load(in_ptr + offsets).to(tl.float32).to(tl.float16)
    tl.store(out_ptr + offset + offsets, val, mask=mask)


@torch.fx.wrap
def triton_fused_cat_to(in_0, in_1, in_2):
    out_B = in_0.size(0) + in_1.size(0) + in_2.size(0)
    out_C = in_0.size(1)
    out_H = in_0.size(2)
    out_W = in_0.size(3)
    
    out = torch.empty((out_B, out_C, out_H, out_W), dtype=torch.float16, device=in_0.device)
    
    BLOCK_SIZE = 1024
    
    offset = 0
    
    # Copy in_2 first
    n2 = in_2.numel()
    if n2 > 0:
        num_programs = (n2 + BLOCK_SIZE - 1) // BLOCK_SIZE
        copy_kernel[(num_programs,)](out, in_2, offset, n2, BLOCK_SIZE)
        offset += n2
    
    # Copy in_1 second
    n1 = in_1.numel()
    if n1 > 0:
        num_programs = (n1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        copy_kernel[(num_programs,)](out, in_1, offset, n1, BLOCK_SIZE)
        offset += n1
    
    # Copy in_0 third
    n0 = in_0.numel()
    if n0 > 0:
        num_programs = (n0 + BLOCK_SIZE - 1) // BLOCK_SIZE
        copy_kernel[(num_programs,)](out, in_0, offset, n0, BLOCK_SIZE)
    
    return out


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.cat([in_2, in_1, in_0], dim = 0)
    tmp_1 = tmp_0.to(dtype = torch.float16)
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_fused_cat_to