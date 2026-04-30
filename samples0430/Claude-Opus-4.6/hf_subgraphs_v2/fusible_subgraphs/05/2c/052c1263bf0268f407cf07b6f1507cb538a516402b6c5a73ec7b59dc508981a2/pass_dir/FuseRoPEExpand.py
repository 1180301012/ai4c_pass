import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    tmp_0 = in_2 * in_1
    return tmp_0


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def mul_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_rope_expand(in_1, in_2):
    out = torch.empty_like(in_2)
    n = in_2.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mul_kernel[grid](in_2, in_1, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return fused_rope_expand