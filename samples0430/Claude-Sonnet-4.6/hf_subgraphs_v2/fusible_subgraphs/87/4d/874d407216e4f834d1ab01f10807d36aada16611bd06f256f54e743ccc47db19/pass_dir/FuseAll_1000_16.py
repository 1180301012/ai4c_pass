import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: full model for bfloat16 and float32 graphs
#   N=1100, C=16, new_zeros((1000, 16))
# Matching the entire computation so the replacement wrapper becomes ONE
# Python-level call, eliminating the GPU idle gap caused by graph breaks.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((1000, 16))
    return (tmp_3, tmp_4, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: broadcast-scale for C=16 (constexpr)
#   out[i, j] = in_1[i] * in_2[i, j]   for j in [0, 16)
# BLOCK_N rows processed per program. Grid is 1-D over N.
# ---------------------------------------------------------------------------

@triton.jit
def _bcast_scale_c16(
    in_1_ptr,             # [N]     – per-row scale weights
    in_2_ptr,             # [N, 16] – feature matrix (contiguous)
    out_ptr,              # [N, 16] – output
    N,
    BLOCK_N: tl.constexpr,
):
    pid    = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_offs = tl.arange(0, 16)                         # [16]
    n_mask = n_offs < N
    mask2d = n_mask[:, None]                           # [BLOCK_N, 1] broadcast

    scale = tl.load(in_1_ptr + n_offs, mask=n_mask, other=0.0)  # [BLOCK_N]
    feat  = tl.load(
        in_2_ptr + n_offs[:, None] * 16 + c_offs[None, :],
        mask=mask2d, other=0.0,
    )  # [BLOCK_N, 16]

    tl.store(
        out_ptr + n_offs[:, None] * 16 + c_offs[None, :],
        scale[:, None] * feat,
        mask=mask2d,
    )


# ---------------------------------------------------------------------------
# Wrapper: all GPU ops in one call → no Python-level graph break between
# the broadcast-multiply kernel and the new_zeros allocation.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fuse_all_1000_16(in_0, in_1, in_2):
    N = in_2.shape[0]            # concrete int at runtime (not SymInt)

    # ── op 1: broadcast multiply (Triton) ─────────────────────────────────
    tmp_1  = torch.empty_like(in_2)
    BLOCK_N = 256
    grid   = ((N + BLOCK_N - 1) // BLOCK_N,)
    _bcast_scale_c16[grid](
        in_1, in_2, tmp_1,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    # ── op 2: zeros allocation (submitted immediately after op 1) ─────────
    tmp_4 = torch.zeros((1000, 16), dtype=in_2.dtype, device=in_2.device)

    # ── op 3: metadata expand – zero GPU cost ─────────────────────────────
    tmp_3 = in_0.view(-1, 1).expand(N, 16)

    return (tmp_3, tmp_4, tmp_1)


def replacement_func():
    return fuse_all_1000_16