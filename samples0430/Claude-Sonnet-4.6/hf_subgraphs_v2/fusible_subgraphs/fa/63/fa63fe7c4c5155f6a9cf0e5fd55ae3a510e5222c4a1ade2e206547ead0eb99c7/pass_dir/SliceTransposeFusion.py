import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Pattern: matches the common in_2[:, :, 1:, :].transpose(-1, -2) subgraph.
# The result (tmp_3) is a non-contiguous view; the subsequent reshape() needs
# a memory copy.  Our kernel fuses slice + transpose into ONE pass writing a
# CONTIGUOUS output, so reshape() and split() downstream become free views.
# ---------------------------------------------------------------------------
def pattern(in_2):
    tmp_2 = in_2[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(in_2):
    return (in_2,)


# ---------------------------------------------------------------------------
# Triton kernel – flat 1-D output indexing.
# Output layout (contiguous): [B, H, K, N_out]
#   flat index = bh * (K * N_out) + k * N_out + n
# Input: in_2[b, h, n+1, k]  (skip row 0)
# Assumes B=1, contiguous in_2 (stride_in=K, stride_ik=1).
# Grid: (B*H, ceil(K*N_out / BLOCK))
# ---------------------------------------------------------------------------
_BLOCK = 512


@triton.jit
def _slice_transpose_kernel(
    in_ptr,
    out_ptr,
    H,
    N,
    K,
    N_out,
    KN_out,
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid    = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < KN_out

    n_idx = offs % N_out
    k_idx = offs // N_out

    # Contiguous in_2: stride_ih = N*K, stride_in = K, stride_ik = 1
    in_offs = pid_bh * (N * K) + (n_idx + 1) * K + k_idx
    data = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid_bh * KN_out + offs, data, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def slice_transpose(in_2):
    B, H, N, K = in_2.shape
    N_out  = N - 1
    KN_out = K * N_out
    out = torch.empty((B, H, K, N_out), dtype=in_2.dtype, device=in_2.device)

    _slice_transpose_kernel[
        (B * H, (KN_out + _BLOCK - 1) // _BLOCK)
    ](
        in_2, out,
        H, N, K, N_out, KN_out,
        BLOCK=_BLOCK,
    )
    return out


def replacement_func():
    return slice_transpose