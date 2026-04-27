import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 2-D tile kernel – plain 2-D grid, BLOCK_N == N so N_tiles == 1 always.
# ---------------------------------------------------------------------------

@triton.jit
def _div_transpose_2d_kernel(
    input_ptr,
    output_ptr,
    M, N, MN, NM,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused (in_0 / scalar).transpose(-1, -2).

    Grid: (BH, ceil(M/BLOCK_M))  –  BLOCK_N == N so N_tiles == 1.
    Input  layout: [BH, M, N]   strides [M*N, N, 1]
    Output layout: [BH, N, M]   strides [N*M, M, 1]
    """
    bh_id  = tl.program_id(0)
    m_tile = tl.program_id(1)

    m_off = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)   # BLOCK_N == N, covers all N

    m_mask = m_off < M

    # Coalesced load [BLOCK_M, BLOCK_N]
    in_ptrs = input_ptr + bh_id * MN + m_off[:, None] * N + n_off[None, :]
    x = tl.load(in_ptrs, mask=m_mask[:, None], other=0.0)
    x = x * scale

    # tl.trans → [BLOCK_N, BLOCK_M]; coalesced store
    out_ptrs = output_ptr + bh_id * NM + n_off[:, None] * M + m_off[None, :]
    tl.store(out_ptrs, tl.trans(x), mask=m_mask[None, :])


_SCALE_1682 = 1.0 / 1.6817928305074292


@torch.fx.wrap
def _fused_div_transpose_1682(in_0):
    B  = in_0.shape[0]
    H  = in_0.shape[1]
    M  = in_0.shape[2]
    N  = in_0.shape[3]
    BH = B * H
    MN = M * N
    NM = N * M

    out = torch.empty(B, H, N, M, dtype=in_0.dtype, device=in_0.device)

    # BLOCK_N = N exactly → N_tiles = 1 → effective 1-D grid over BH
    if N <= 8:
        BLOCK_N = 8
    else:
        BLOCK_N = 64

    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    elif M <= 128:
        BLOCK_M = 128
    else:
        BLOCK_M = 256

    m_tiles = (M + BLOCK_M - 1) // BLOCK_M

    _div_transpose_2d_kernel[(BH, m_tiles)](
        in_0, out,
        M, N, MN, NM,
        _SCALE_1682,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_div_transpose_1682