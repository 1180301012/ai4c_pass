import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = x.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_4


def replacement_args(x):
    return (x,)


@triton.jit
def _mean_view_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    base = c * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    mean_val = (total / HW).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + c, mean_val)


@torch.fx.wrap
def _mean_view_wrapper(x):
    C = x.shape[1]
    HW = x.numel() // C

    out = torch.empty((1, 1, C), dtype=x.dtype, device=x.device)

    # Small HW: max throughput with large-block, small-iter design
    # BLOCK_HW=64, num_warps=4 → 128 threads, 0.5 elements/thread
    if HW <= 128:
        _mean_view_kernel[(C,)](x, out, HW, BLOCK_HW=64,  num_warps=4)
    elif HW <= 256:
        _mean_view_kernel[(C,)](x, out, HW, BLOCK_HW=64,  num_warps=4)
    elif HW <= 512:
        _mean_view_kernel[(C,)](x, out, HW, BLOCK_HW=256, num_warps=4)
    elif HW <= 1024:
        _mean_view_kernel[(C,)](x, out, HW, BLOCK_HW=512, num_warps=8)
    elif HW <= 2048:
        _mean_view_kernel[(C,)](x, out, HW, BLOCK_HW=1024, num_warps=16)
    else:
        _mean_view_kernel[(C,)](x, out, HW, BLOCK_HW=4096, num_warps=32)

    return out


def replacement_func():
    return _mean_view_wrapper