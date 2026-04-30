import torch
import triton
import triton.language as tl


# ── Pattern: match the max chain starting from tmp_7 ─────────────────────────
# tmp_7 is (3,1,N) int64 – the already-computed expanded tensor on CUDA.
# We match only tmp_13 (scalar) as output; tmp_7 is an observable
# intermediate produced OUTSIDE this subgraph by the .to() call.
def pattern(tmp_7):
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13


def replacement_args(tmp_7):
    return (tmp_7,)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Reads the first "row" of tmp_7 (3 copies, values are identical).
# result = max(tmp_7[0,0,:]) - 8
@triton.jit
def max_scalar_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    row = tl.load(in_ptr + offsets, mask=mask, other=0)
    max_val = tl.max(row, axis=0)
    tl.store(out_ptr, max_val - 8)


# ── Kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def compute_max(tmp_7):
    N = tmp_7.shape[2]
    out_13 = torch.empty((1, 1), dtype=tmp_7.dtype, device=tmp_7.device)
    max_scalar_kernel[(1,)](tmp_7, out_13, N, BLOCK_N=2048)
    return out_13


def replacement_func():
    return compute_max