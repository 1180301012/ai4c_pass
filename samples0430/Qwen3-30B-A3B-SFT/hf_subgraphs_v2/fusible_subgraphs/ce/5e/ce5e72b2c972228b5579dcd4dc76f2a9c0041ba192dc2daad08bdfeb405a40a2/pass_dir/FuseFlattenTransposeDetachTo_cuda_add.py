import torch
import triton
import triton.language as tl
from torch import device


def pattern(conv_out, pos_emb):
    """
    Match: flatten(2) -> transpose(1,2) -> detach -> type_as -> to(cuda) -> add
    conv_out: [B, C, D1, D2, D3]  (raw output of conv3d on CUDA)
    pos_emb:  [B, N, C]  (position embeddings, already on CUDA)
    """
    tmp_4 = conv_out.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = pos_emb.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv_out, pos_emb):
    return (conv_out, pos_emb)


# ---------------------------------------------------------------------------
# Fused 2D-tiled kernel: transpose conv_out[C,N] + pos_emb[N,C] → output[N,C]
#
# conv_out has shape [B, C, D1, D2, D3] with strides [C*N, N, D2*D3, D3, 1]
# where N = D1*D2*D3.  Element [0, c, n] (n = h*D2*D3 + w*D3 + t):
#   flat offset = c * N + n   (same as for a contiguous [C, N] view)
#
# pos_emb  has shape [B, N, C]  contiguous → offset = n*C + c
# output   has shape [B, N, C]  contiguous → offset = n*C + c
#
# 2-D grid (BLOCK_N along N, BLOCK_C along C).
# tl.trans() gives coalesced reads of the conv_out tile.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_C': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C'],
)
@triton.jit
def _transpose_add_kernel(
    conv_ptr,
    emb_ptr,
    out_ptr,
    N,              # total spatial = D1*D2*D3  (e.g. 1568)
    C,              # channels                   (e.g. 768)
    BLOCK_N: tl.constexpr,   # tile rows (N dim)
    BLOCK_C: tl.constexpr,   # tile cols (C dim)
):
    """
    Each program computes output[pid_n*BN:(pid_n+1)*BN, pid_c*BC:(pid_c+1)*BC]:
        out[n, c] = conv_out[c, n] + emb[n, c]

    Loads conv_out as [BC, BN] (coalesced in N direction) then tl.trans to [BN, BC].
    Loads pos_emb as [BN, BC] (coalesced in C direction).
    Writes output[BN, BC] (coalesced in C direction).
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n0 = pid_n * BLOCK_N
    c0 = pid_c * BLOCK_C

    n_idx = n0 + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_idx = c0 + tl.arange(0, BLOCK_C)   # [BLOCK_C]

    # ---- Load conv_out[c, n] → [BLOCK_C, BLOCK_N], coalesced in N (fast dim) ----
    x_off = c_idx[:, None] * N + n_idx[None, :]   # [BC, BN]
    mask_x = (c_idx[:, None] < C) & (n_idx[None, :] < N)
    x = tl.load(conv_ptr + x_off, mask=mask_x, other=0.0)

    # ---- Load pos_emb[n, c] → [BLOCK_N, BLOCK_C], coalesced in C (fast dim) ----
    y_off = n_idx[:, None] * C + c_idx[None, :]   # [BN, BC]
    mask_y = (n_idx[:, None] < N) & (c_idx[None, :] < C)
    y = tl.load(emb_ptr + y_off, mask=mask_y, other=0.0)

    # ---- Fused add with register-level transpose ----
    result = tl.trans(x) + y   # [BN, BC]

    # ---- Store output[n, c] ----
    out_off = n_idx[:, None] * C + c_idx[None, :]   # [BN, BC]
    mask_out = (n_idx[:, None] < N) & (c_idx[None, :] < C)
    tl.store(out_ptr + out_off, result, mask=mask_out)


@torch.fx.wrap
def fused_flatten_transpose_add(conv_out, pos_emb):
    """
    Fused replacement for:
        flatten(2) → transpose(1,2) → detach → type_as → to_cuda → add

    conv_out : [B, C, D1, D2, D3]  on CUDA  (raw 5-D conv3d output)
    pos_emb  : [B, N, C]  (already on CUDA per debug output)
    returns  : [B, N, C]  on CUDA
    """
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    # Compute N = D1*D2*D3 from the 5-D conv output shape
    N = 1
    for i in range(2, len(conv_out.shape)):
        N *= conv_out.shape[i]

    # pos_emb is already on CUDA (confirmed by debug); .to() is a no-op here
    pos_emb_cuda = pos_emb.to(device=conv_out.device, dtype=conv_out.dtype)

    # Allocate output [B, N, C] contiguous
    output = torch.empty_like(pos_emb_cuda)

    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(C, meta['BLOCK_C']),
        B,
    )

    _transpose_add_kernel[grid](
        conv_out,
        pos_emb_cuda,
        output,
        N,
        C,
    )

    return output


def replacement_func():
    return fused_flatten_transpose_add