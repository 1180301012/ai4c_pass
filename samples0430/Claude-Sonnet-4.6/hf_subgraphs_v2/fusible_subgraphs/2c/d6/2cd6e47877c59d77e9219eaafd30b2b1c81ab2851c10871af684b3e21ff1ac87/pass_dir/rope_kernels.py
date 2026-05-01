"""
Shared Triton kernels + dispatch function for EVA-02 RoPE.

All four pass files (RoPE_6_256, RoPE_12_196, RoPE_12_256, RoPE_16_256) import
rope_dispatch from this module so that replacement_func() returns the SAME
function object across all passes, satisfying output_pass_replacement_func_limit.

NH and N are inferred dynamically from in_3.shape so no route string is needed.

Two computation paths per forward call:
  Path 1 (query): out = x * cos + rotate_half(x) * sin, prepend cls token
  Path 2 (key):   out = k * sin + rotate_half(k) * cos (using pos_embed halves), prepend k_cls

rotate_half: rotate_half(x)[2j] = -x[2j+1], rotate_half(x)[2j+1] = x[2j]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: copy CLS from in_2 [1, nH, 1, D] to out1 [1, nH, N+1, D] position 0
# ---------------------------------------------------------------------------
@triton.jit
def _copy_cls_q(
    cls_ptr,
    out_ptr,
    N_PLUS_1,
    HALF_D: tl.constexpr,
):
    h = tl.program_id(0)
    D = HALF_D * 2
    d_e = tl.arange(0, HALF_D) * 2
    d_o = d_e + 1
    v_e = tl.load(cls_ptr + h * D + d_e)
    v_o = tl.load(cls_ptr + h * D + d_o)
    tl.store(out_ptr + h * N_PLUS_1 * D + d_e, v_e)
    tl.store(out_ptr + h * N_PLUS_1 * D + d_o, v_o)


# ---------------------------------------------------------------------------
# Kernel: copy CLS from in_4 [1, nH, N+1, D] position 0 to out2 position 0
# ---------------------------------------------------------------------------
@triton.jit
def _copy_cls_k(
    k_ptr,
    out_ptr,
    N_PLUS_1,
    HALF_D: tl.constexpr,
):
    h = tl.program_id(0)
    D = HALF_D * 2
    d_e = tl.arange(0, HALF_D) * 2
    d_o = d_e + 1
    base = h * N_PLUS_1 * D
    v_e = tl.load(k_ptr + base + d_e)
    v_o = tl.load(k_ptr + base + d_o)
    tl.store(out_ptr + base + d_e, v_e)
    tl.store(out_ptr + base + d_o, v_o)


# ---------------------------------------------------------------------------
# Kernel: Path 1 RoPE — one program per (head, position)
# ---------------------------------------------------------------------------
@triton.jit
def _rope_p1(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    N, N_PLUS_1,
    HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid // N
    n = pid % N
    D = HALF_D * 2
    d_e = tl.arange(0, HALF_D) * 2
    d_o = d_e + 1
    x_base   = (h * N + n) * D
    emb_base = n * D
    out_base = h * N_PLUS_1 * D + (n + 1) * D
    x_e = tl.load(x_ptr   + x_base   + d_e)
    x_o = tl.load(x_ptr   + x_base   + d_o)
    c_e = tl.load(cos_ptr + emb_base + d_e)
    c_o = tl.load(cos_ptr + emb_base + d_o)
    s_e = tl.load(sin_ptr + emb_base + d_e)
    s_o = tl.load(sin_ptr + emb_base + d_o)
    tl.store(out_ptr + out_base + d_e, x_e * c_e - x_o * s_e)
    tl.store(out_ptr + out_base + d_o, x_o * c_o + x_e * s_o)


# ---------------------------------------------------------------------------
# Kernel: Path 2 RoPE — one program per (head, position)
# ---------------------------------------------------------------------------
@triton.jit
def _rope_p2(
    k_ptr, pos_ptr, out_ptr,
    N, N_PLUS_1,
    HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid // N
    n = pid % N
    D = HALF_D * 2
    D2 = D * 2
    d_e = tl.arange(0, HALF_D) * 2
    d_o = d_e + 1
    k_base   = h * N_PLUS_1 * D + (n + 1) * D
    pos_base = n * D2
    out_base = h * N_PLUS_1 * D + (n + 1) * D
    k_e   = tl.load(k_ptr   + k_base   + d_e)
    k_o   = tl.load(k_ptr   + k_base   + d_o)
    cos_e = tl.load(pos_ptr + pos_base + d_e)
    cos_o = tl.load(pos_ptr + pos_base + d_o)
    sin_e = tl.load(pos_ptr + pos_base + D + d_e)
    sin_o = tl.load(pos_ptr + pos_base + D + d_o)
    tl.store(out_ptr + out_base + d_e, k_e * sin_e - k_o * cos_e)
    tl.store(out_ptr + out_base + d_o, k_o * sin_o + k_e * cos_o)


# ---------------------------------------------------------------------------
# Shared dispatch — returned by replacement_func() in ALL pass files.
# NH and N are inferred from in_3.shape so no route/string argument is needed.
# D is always 64 for these graphs.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def rope_dispatch(t0, t1, t2, t3, t4, route):
    """
    Unified dispatch for both RoPE paths (single return value each).
    route = "q_NH_N" | "k_NH_N"
    Path q (query):  t0=x, t1=cos, t2=sin, t3=cls_q, t4=ref
    Path k (key):    t0=k, t1=pos,  t2=ref, t3=ref,   t4=ref
    """
    parts = route.split("_")
    path  = parts[0]
    NH    = int(parts[1])
    N     = int(parts[2])
    N1    = N + 1
    HD    = 32
    D     = 64

    out = torch.empty((1, NH, N1, D), dtype=t4.dtype, device=t4.device)

    if path == "q":
        _copy_cls_q[(NH,)](t3, out, N1, HALF_D=HD)
        _rope_p1[(NH * N,)](t0, t1, t2, out, N, N1, HALF_D=HD)
    else:
        _copy_cls_k[(NH,)](t0, out, N1, HALF_D=HD)
        _rope_p2[(NH * N,)](t0, t1, out, N, N1, HALF_D=HD)

    return out