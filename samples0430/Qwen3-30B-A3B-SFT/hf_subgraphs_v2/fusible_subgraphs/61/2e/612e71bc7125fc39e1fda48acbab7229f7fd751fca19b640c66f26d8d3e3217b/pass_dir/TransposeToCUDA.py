import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Pattern: matches  in_0.t().to(device(type='cuda'))
#   tmp_2 = in_0.t()
#   tmp_3 = tmp_2.to(device(type='cuda'))
#   return tmp_3
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: out-of-place [1, N] -> [N, 1] copy (transpose + to-cuda)
#   Both tensors share the same flat 1-D layout; the operation is just a
#   copy where out[col, 0] = in[0, col].
# ---------------------------------------------------------------------------
@triton.jit
def _transpose_1n_to_n1_kernel(
    in0_ptr,
    out1_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    # in_0 is [1, N] contiguous; element [0, col] is at flat offset col
    x = tl.load(in0_ptr + offs, mask=mask, other=0.0)
    # out1 is [N, 1] contiguous; element [col, 0] is at flat offset col
    tl.store(out1_ptr + offs, x, mask=mask)


@torch.fx.wrap
def transpose_to_cuda(in_0):
    # in_0: [1, N] on any device
    N = in_0.shape[1]
    out = torch.empty((N, 1), dtype=in_0.dtype, device=in_0.device)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_N = max(BLOCK_N, 128)
    grid = (1,)

    _transpose_1n_to_n1_kernel[grid](
        in_0, out,
        N,
        BLOCK_N=BLOCK_N,
    )
    return out


def replacement_func():
    return transpose_to_cuda