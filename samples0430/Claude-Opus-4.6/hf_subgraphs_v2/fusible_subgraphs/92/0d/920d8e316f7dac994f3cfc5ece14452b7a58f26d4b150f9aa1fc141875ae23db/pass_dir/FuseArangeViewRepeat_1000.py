import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = torch.ops.aten.view.default(x, [1, -1])
    tmp_2 = torch.ops.aten.repeat.default(tmp_1, [2, 1])
    return (tmp_2,)


def replacement_args(x):
    return (x,)


@triton.jit
def repeat_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load from input
    vals = tl.load(in_ptr + offsets, mask=mask)
    # Store to row 0 and row 1
    tl.store(out_ptr + offsets, vals, mask=mask)
    tl.store(out_ptr + N + offsets, vals, mask=mask)


@torch.fx.wrap
def view_repeat_optimized(x):
    N = x.numel()
    out = torch.empty((2, N), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    repeat_kernel[grid](x, out, N, BLOCK_SIZE)
    return (out,)


def replacement_func():
    return view_repeat_optimized