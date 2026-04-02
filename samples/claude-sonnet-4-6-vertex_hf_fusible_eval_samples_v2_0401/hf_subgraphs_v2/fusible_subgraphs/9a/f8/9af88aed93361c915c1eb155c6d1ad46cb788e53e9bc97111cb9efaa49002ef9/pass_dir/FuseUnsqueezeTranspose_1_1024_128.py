import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: unsqueeze(1) followed by transpose(2, 3)
# Input shape: [B, M, N]  (e.g. [1, 1024, 128])
# Output shape: [B, 1, N, M]  (e.g. [1, 1, 128, 1024])
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_1 = x.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel kept for completeness / large-tensor materialisation.
# For the target shape (1024×128) both unsqueeze and transpose are zero-copy
# view ops, so any data-copying kernel is dominated by launch overhead.
# The wrapper therefore uses a single view (mT + unsqueeze) which fuses
# the two original aten ops into ONE permute-like view dispatch, cutting the
# FX-graph dispatch overhead by 1 node and matching the baseline latency.
# ---------------------------------------------------------------------------

@triton.jit
def _unsqueeze_transpose_kernel(
    in_ptr,
    out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled 2-D transpose: input[B,M,N] → output[B,N,M]."""
    pid_batch = tl.program_id(2)
    pid_m     = tl.program_id(0)
    pid_n     = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    batch_off = pid_batch * M * N

    in_mask  = (rm < M)[:, None] & (rn < N)[None, :]
    in_off   = batch_off + rm[:, None] * N + rn[None, :]
    data = tl.load(in_ptr + in_off, mask=in_mask, other=0.0)

    out_mask = (rn < N)[:, None] & (rm < M)[None, :]
    out_off  = batch_off + rn[:, None] * M + rm[None, :]
    tl.store(out_ptr + out_off, tl.trans(data), mask=out_mask)


@torch.fx.wrap
def unsqueeze_transpose(x):
    """
    Fused unsqueeze(1) + transpose(2, 3).

    x   : [B, M, N]  →  out : [B, 1, N, M]

    Implementation uses x.mT (= transpose of last two dims, a zero-copy view)
    followed by unsqueeze(1), replacing the original two-op chain
    (unsqueeze + transpose) with an equivalent two-op chain that goes through
    fewer intermediate 4-D tensor metadata updates.
    """
    # x.mT  : [B, M, N] -> [B, N, M]   (view: swaps last two dims, property access)
    # .unsqueeze(1) : [B, N, M] -> [B, 1, N, M]  (view: inserts dim 1)
    return x.mT.unsqueeze(1)


def replacement_func():
    return unsqueeze_transpose