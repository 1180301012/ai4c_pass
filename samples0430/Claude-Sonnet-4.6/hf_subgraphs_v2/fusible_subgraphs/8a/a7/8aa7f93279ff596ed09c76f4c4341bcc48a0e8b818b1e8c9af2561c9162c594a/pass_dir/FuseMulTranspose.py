import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------
# Pass: replace  x * 0.1767766952966369  with an optimised Triton kernel.
#
# Shape is [70, 1, 49, 32] = 109 760 float16/bfloat16 elements.
# Kernel uses 1-D grid; BLOCK_SIZE=1024 gives 108 blocks → ~2 blocks/SM
# on the A30, with each thread doing 8 float16 elements (128-bit loads).
# -----------------------------------------------------------------------

def pattern(x):
    return x * 0.1767766952966369


def replacement_args(x):
    return (x,)


@triton.jit
def _scale_kernel(
    x_ptr, out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x    = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scalar, mask=mask)


@torch.fx.wrap
def _scale_mul(x):
    n   = x.numel()
    out = torch.empty_like(x)
    # 108 blocks of 1024 elements → ~2 blocks/SM on A30
    # Triton uses default num_warps=4 → 128 threads, 8 elements/thread (128-bit loads)
    n_blocks = (n + 1023) // 1024
    _scale_kernel[(n_blocks,)](x, out, 0.1767766952966369, n, BLOCK_SIZE=1024)
    return out


def replacement_func():
    return _scale_mul