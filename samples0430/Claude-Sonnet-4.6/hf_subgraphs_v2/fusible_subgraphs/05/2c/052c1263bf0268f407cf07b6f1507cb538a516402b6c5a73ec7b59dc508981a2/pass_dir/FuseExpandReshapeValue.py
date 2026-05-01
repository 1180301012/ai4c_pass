import torch
import triton
import triton.language as tl


def pattern(x):
    tmp = x[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp2 = tmp.expand(1, 1, 8, 3, 256)
    tmp3 = tmp2.reshape(1, 8, 3, 256)
    return tmp3


def replacement_args(x):
    return (x,)


@triton.jit
def _expand_direct_kernel(
    src_ptr, dst_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    # broadcast to 8 heads: dst[h*N + offset] = val
    tl.store(dst_ptr + 0 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 1 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 2 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 3 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 4 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 5 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 6 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 7 * N + offsets, val, mask=mask)


@torch.fx.wrap
def _expand_wrapper(x):
    N = 768          # 1*1*3*256
    SEQ_LEN = 3
    DIM = 256
    BLOCK_SIZE = 256
    out = torch.empty((1, 8, SEQ_LEN, DIM), dtype=x.dtype, device=x.device)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _expand_direct_kernel[grid](x, out, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return _expand_wrapper