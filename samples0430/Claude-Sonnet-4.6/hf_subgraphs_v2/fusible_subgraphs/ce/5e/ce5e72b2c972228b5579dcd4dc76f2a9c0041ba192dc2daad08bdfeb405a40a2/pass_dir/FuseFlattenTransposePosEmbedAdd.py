import torch
import triton
import triton.language as tl
from torch import device


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32,  'BLOCK_C': 32},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_C': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_C': 32},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_N': 32,  'BLOCK_C': 64},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 32},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 32,  'BLOCK_C': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_C': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 32,  'BLOCK_C': 32},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_C': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_C': 128}, num_warps=8,  num_stages=3),
    ],
    key=['N', 'C'],
)
@triton.jit
def _fused_transpose_add_kernel(
    x_ptr,    # [B, C, N]  conv output (contiguous: stride_xc=N, stride_xn=1)
    pos_ptr,  # [B, N, C]  position embeddings (contiguous: stride_pn=C, stride_pc=1)
    out_ptr,  # [B, N, C]  output
    B, N, C,
    stride_xb, stride_xc, stride_xn,
    stride_pb, stride_pn, stride_pc,
    stride_ob, stride_on, stride_oc,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Coalesced transpose-add:
      1. Load x in [BLOCK_C, BLOCK_N] order → N is the fast dim → coalesced reads.
      2. tl.trans() → register-only flip to [BLOCK_N, BLOCK_C].
      3. Load pos and store out in [BLOCK_N, BLOCK_C] → C is the fast dim → coalesced.
    Grid: (ceil(N/BLOCK_N), ceil(C/BLOCK_C), B)
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    c_start = pid_c * BLOCK_C

    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_offs = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]

    # ---- Load x[b, c, n] as [BLOCK_C, BLOCK_N] --------------------------------
    # fast dim = N (stride_xn=1) → each warp reads contiguous n values → coalesced
    mask_x = (c_offs[:, None] < C) & (n_offs[None, :] < N)   # [BLOCK_C, BLOCK_N]
    x_idx = (pid_b * stride_xb
              + c_offs[:, None] * stride_xc
              + n_offs[None, :] * stride_xn)                   # [BLOCK_C, BLOCK_N]
    x_tile = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)   # [BLOCK_C, BLOCK_N]

    # ---- Register-only transpose: [BLOCK_C, BLOCK_N] → [BLOCK_N, BLOCK_C] ----
    x_t = tl.trans(x_tile)                                     # [BLOCK_N, BLOCK_C]

    # ---- Load pos[b, n, c] and write out[b, n, c] as [BLOCK_N, BLOCK_C] ------
    # fast dim = C (stride_pc=1) → coalesced reads & writes
    mask_out = (n_offs[:, None] < N) & (c_offs[None, :] < C)  # [BLOCK_N, BLOCK_C]

    pos_idx = (pid_b * stride_pb
               + n_offs[:, None] * stride_pn
               + c_offs[None, :] * stride_pc)                  # [BLOCK_N, BLOCK_C]
    pos_tile = tl.load(pos_ptr + pos_idx, mask=mask_out, other=0.0)

    out_tile = x_t + pos_tile

    out_idx = (pid_b * stride_ob
               + n_offs[:, None] * stride_on
               + c_offs[None, :] * stride_oc)                  # [BLOCK_N, BLOCK_C]
    tl.store(out_ptr + out_idx, out_tile, mask=mask_out)


# Module-level cache: avoids repeated CPU→GPU copies of the static pos_emb parameter
_pos_emb_cuda_cache: dict = {}


@torch.fx.wrap
def fused_flatten_transpose_add(conv_out, pos_emb):
    """
    Replaces: flatten(2) + transpose(1,2) + detach + type_as + to(cuda) + add
    conv_out : [B, C, T, H, W]  on CUDA  (5-D conv3d output, always contiguous)
    pos_emb  : [B, N, C]         may be on CPU (static model parameter)
    returns  : [B, N, C]         on CUDA
    """
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    # N = T * H * W — pure Python int arithmetic, no aten dispatch
    N = conv_out.shape[2] * conv_out.shape[3] * conv_out.shape[4]

    # Transfer pos_emb to CUDA with the correct dtype.
    # torch.as_tensor is explicitly whitelisted by the framework.
    pos_emb_cuda = torch.as_tensor(pos_emb, dtype=conv_out.dtype, device=conv_out.device)

    # Allocate output (whitelisted factory op)
    out = torch.empty((B, N, C), dtype=conv_out.dtype, device=conv_out.device)

    # Logical strides for conv_out viewed as [B, C, N] (always contiguous after conv3d)
    stride_xb = C * N
    stride_xc = N       # channel stride = T*H*W
    stride_xn = 1       # spatial stride = 1 (innermost spatial dim)

    # pos_emb_cuda and out are [B, N, C] contiguous
    stride_bn = N * C
    stride_n  = C
    stride_c  = 1

    grid = lambda meta: (
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        (C + meta['BLOCK_C'] - 1) // meta['BLOCK_C'],
        B,
    )

    _fused_transpose_add_kernel[grid](
        conv_out, pos_emb_cuda, out,
        B, N, C,
        stride_xb, stride_xc, stride_xn,
        stride_bn, stride_n,  stride_c,   # pos strides
        stride_bn, stride_n,  stride_c,   # out strides
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(conv_out, pos_emb):
    tmp_4 = conv_out.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = pos_emb.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv_out, pos_emb):
    return (conv_out, pos_emb)


def replacement_func():
    return fused_flatten_transpose_add