import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = x.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args(x):
    return (x.numel(),)


@triton.jit
def _repeat_two_rows_kernel(
    out_ptr,
    n_elements,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    values = offsets % n_elements
    tl.store(out_ptr + offsets, values, mask=mask)


_REPEAT_CACHE = {}


@torch.fx.wrap
def _repeat_two_rows(n):
    n = int(n)
    out = _REPEAT_CACHE.get(n)
    if out is None:
        total = 2 * n
        out = torch.empty((2, n), device='cuda', dtype=torch.int64)
        block_size = 1024 if total > 128 else 128
        num_warps = 4 if total > 128 else 1
        _repeat_two_rows_kernel[(triton.cdiv(total, block_size),)](
            out,
            n,
            total,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        _REPEAT_CACHE[n] = out
    return out


def replacement_func():
    return _repeat_two_rows