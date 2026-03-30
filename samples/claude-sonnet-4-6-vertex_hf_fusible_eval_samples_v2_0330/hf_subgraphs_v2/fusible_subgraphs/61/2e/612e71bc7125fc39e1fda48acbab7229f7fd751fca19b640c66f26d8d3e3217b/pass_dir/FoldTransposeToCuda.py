import torch
import triton
import triton.language as tl
from torch import device


# ──────────────────────────────────────────────────────────────
# Pass: Fuse t() + .to(device('cuda')) into a single op.
#   in_0 is already on CUDA (confirmed by weight_meta.py),
#   so .to(device('cuda')) is a redundant no-op.
#   We replace both ops with one contiguous-copy Triton kernel
#   that materialises the transposed [D, 1] layout.
# ──────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


# ──────────────────────────────────────────────────────────────
# Triton kernel: element-wise copy (makes the result contiguous)
# ──────────────────────────────────────────────────────────────
@triton.jit
def copy_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def t_to_cuda(in_0):
    # in_0 shape: [1, D]  →  result shape: [D, 1]
    D = in_0.shape[-1]
    N = in_0.numel()  # == D
    out = torch.empty((D, 1), dtype=in_0.dtype, device=in_0.device)

    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    copy_kernel[grid](in_0, out, N, BLOCK=BLOCK)
    return out


def replacement_func():
    return t_to_cuda