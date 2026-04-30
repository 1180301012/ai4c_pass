import torch
import triton
import triton.language as tl


def pattern(x, y, z):
    out = torch.stack([x, y, z])
    return out


def replacement_args(x, y, z):
    return (x, y, z)


@triton.jit
def copy_with_base_kernel(
    src_ptr,
    dst_ptr,
    total_elems,
    base_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elems
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + base_offset + offs, x, mask=mask)


@torch.fx.wrap
def stack3_triton(x, y, z):
    out = torch.empty((3,) + tuple(x.shape), device=x.device, dtype=x.dtype)
    total = x.numel()
    block_size = 1024
    grid = (triton.cdiv(total, block_size),)
    copy_with_base_kernel[grid](x, out, total, 0, BLOCK_SIZE=block_size, num_warps=8)
    copy_with_base_kernel[grid](y, out, total, total, BLOCK_SIZE=block_size, num_warps=8)
    copy_with_base_kernel[grid](z, out, total, 2 * total, BLOCK_SIZE=block_size, num_warps=8)
    return out


def replacement_func():
    return stack3_triton