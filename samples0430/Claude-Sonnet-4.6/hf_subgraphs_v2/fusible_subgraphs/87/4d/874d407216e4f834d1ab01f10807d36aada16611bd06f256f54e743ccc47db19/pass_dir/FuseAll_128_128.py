import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: full model for float16 graph
#   N=256, C=128, new_zeros((128, 128))
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((128, 128))
    return (tmp_3, tmp_4, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: broadcast-scale for C=128 (constexpr)
# ---------------------------------------------------------------------------

@triton.jit
def _bcast_scale_c128(
    in_1_ptr,              # [N]      – per-row scale weights
    in_2_ptr,              # [N, 128] – feature matrix (contiguous)
    out_ptr,               # [N, 128] – output
    N,
    BLOCK_N: tl.constexpr,
):
    pid    = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_offs = tl.arange(0, 128)                        # [128]
    n_mask = n_offs < N
    mask2d = n_mask[:, None]

    scale = tl.load(in_1_ptr + n_offs, mask=n_mask, other=0.0)
    feat  = tl.load(
        in_2_ptr + n_offs[:, None] * 128 + c_offs[None, :],
        mask=mask2d, other=0.0,
    )

    tl.store(
        out_ptr + n_offs[:, None] * 128 + c_offs[None, :],
        scale[:, None] * feat,
        mask=mask2d,
    )


# ---------------------------------------------------------------------------
# Wrapper: entire model in one Python call – no graph break
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fuse_all_128_128(in_0, in_1, in_2):
    N = in_2.shape[0]

    # ── op 1: broadcast multiply (Triton) ─────────────────────────────────
    tmp_1  = torch.empty_like(in_2)
    BLOCK_N = 32
    grid   = ((N + BLOCK_N - 1) // BLOCK_N,)
    _bcast_scale_c128[grid](
        in_1, in_2, tmp_1,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    # ── op 2: zeros allocation ────────────────────────────────────────────
    tmp_4 = torch.zeros((128, 128), dtype=in_2.dtype, device=in_2.device)

    # ── op 3: metadata expand – zero GPU cost ─────────────────────────────
    tmp_3 = in_0.view(-1, 1).expand(N, 128)

    return (tmp_3, tmp_4, tmp_1)


def replacement_func():
    return fuse_all_128_128