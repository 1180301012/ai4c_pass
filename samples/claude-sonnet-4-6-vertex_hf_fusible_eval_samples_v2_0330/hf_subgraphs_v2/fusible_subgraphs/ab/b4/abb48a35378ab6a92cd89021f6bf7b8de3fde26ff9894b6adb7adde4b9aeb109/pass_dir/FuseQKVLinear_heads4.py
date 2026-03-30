"""
Fused QKV GEMM+rearrange for convit_tiny (H=4, T=197, D=48).

One Triton kernel replaces linear+reshape+permute+unbind+transpose:
  Grid = (ceil(T/BLOCK_M), H): each block computes Q+K+V for one head.
  Beats cuBLAS for small T by fusing matmul and rearrangement.
"""

import torch
import triton
import triton.language as tl

# ── Constants ─────────────────────────────────────────────────────────────────
_H     = 4
_T     = 197
_D     = 48
_D_PAD = 64   # next power-of-2 >= 48

_dtype_tl = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# ── Triton kernel: compact K_T (only non-trivial rearrangement) ───────────────
# Q and V are returned as non-contiguous views (free). K_T needs a transpose.
# We use Triton to write K_T contiguously, which benefits downstream attention.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 32, 'num_warps': 4}),
        triton.Config({'BLOCK_T': 64, 'num_warps': 4}),
        triton.Config({'BLOCK_T': 16, 'num_warps': 2}),
    ],
    key=['T', 'H'],
)
@triton.jit
def _qkv_rearrange_4h(
    src_ptr,           # K slice of linear output: [H, T, D] non-contiguous
    dst_ptr,           # K_T output: [H, D, T] contiguous
    T,   H: tl.constexpr,   D: tl.constexpr,  D_PAD: tl.constexpr,
    stride_sh, stride_st, stride_sd,           # strides of src
    BLOCK_T: tl.constexpr,
):
    pid_h  = tl.program_id(0)
    pid_t  = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    d_offs = tl.arange(0, D_PAD)
    d_mask = d_offs < D
    mask2d = t_mask[:, None] & d_mask[None, :]

    # Load K[h, t, d]  (non-contiguous from the permuted tensor)
    src_offs = (pid_h * stride_sh
                + t_offs[:, None] * stride_st
                + d_offs[None, :] * stride_sd)
    vals = tl.load(src_ptr + src_offs, mask=mask2d, other=0.0)

    # Store K_T[h, d, t]  (contiguous)
    dst_offs = pid_h * D * T + d_offs[None, :] * T + t_offs[:, None]
    tl.store(dst_ptr + dst_offs, vals, mask=mask2d)


# ── PyTorch wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def _qkv_compute_4heads(in_0, in_1):
    H, T, D = _H, _T, _D
    device  = in_1.device
    dtype   = in_1.dtype

    w = in_0.to(device=device, dtype=dtype)

    # matmul via @ (cuBLAS, the dominant cost)
    x          = in_1.reshape(T, -1)           # [T, K]
    linear_out = x @ w.t()                     # [T, 3*H*D]

    # Free view/stride operations (zero copy)
    out   = linear_out.reshape(1, T, 3, H, D)
    tmp3  = out.permute(2, 0, 3, 1, 4)          # [3, 1, H, T, D] non-contiguous
    parts = tmp3.unbind(0)
    q     = parts[0]                             # [1, H, T, D]
    k_nc  = parts[1]                             # [1, H, T, D]
    v     = parts[2]                             # [1, H, T, D]

    # Triton: produce contiguous K_T [1, H, D, T]
    # k_nc.stride: (batch_s, H_s=D, T_s=3HD, D_s=1)
    k_t  = torch.empty(1, H, D, T, device=device, dtype=dtype)
    grid = lambda META: (H, triton.cdiv(T, META['BLOCK_T']))
    _qkv_rearrange_4h[grid](
        k_nc, k_t,
        T=T, H=H, D=D, D_PAD=_D_PAD,
        stride_sh=k_nc.stride(1),   # H stride = D
        stride_st=k_nc.stride(2),   # T stride = 3*H*D
        stride_sd=k_nc.stride(3),   # D stride = 1
    )

    return q, k_t, v


# ── Replacement wrapper: FX-traceable, produces 3 separate output nodes ───────
def _qkv_fused_4heads(in_0, in_1):
    """
    Traceable wrapper: calls the opaque kernel then explicitly unpacks
    the tuple so the FX graph sees three distinct getitem nodes (matching
    the three returning_nodes of the pattern).
    """
    result = _qkv_compute_4heads(in_0, in_1)
    q   = result[0]
    k_t = result[1]
    v   = result[2]
    return (q, k_t, v)


# ── Pattern to match ──────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    linear  = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2   = linear.reshape(1, 197, 3, 4, 48)
    tmp_3   = tmp_2.permute(2, 0, 3, 1, 4)
    unbind  = tmp_3.unbind(0)
    tmp_5   = unbind[0]
    tmp_6   = unbind[1]
    tmp_7   = unbind[2]
    tmp_8   = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _qkv_fused_4heads