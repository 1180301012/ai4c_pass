import torch
import triton
import triton.language as tl


# ── Paired kernel: each program handles ONE (h, s_out) row for BOTH Q & K ───
# Grid = H * Sp1  (half the programs of the split Q/K approach).
# Q and K share the program → better SM utilisation, one launch for both.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S', 'Sp1'],
)
@triton.jit
def _rope_tail_paired(
    # ── Q inputs ──
    q_cls_ptr,      # in_2       : [1, H,  1,  D]  contiguous
    q_ptr,          # in_3       : [1, H,  S,  D]  contiguous
    cos_ptr,        # in_1       : [S, D]           contiguous
    reshape_q_ptr,  # reshape_q  : [1, H,  S,  D]  contiguous
    sin_ptr,        # in_5       : [S, D]           contiguous
    q_out_ptr,      # q output   : [1, H, Sp1, D]  contiguous
    # ── K inputs ──
    k_ptr,          # in_4       : [1, H, Sp1, D]  contiguous  (cls comes from here)
    in0_ptr,        # in_0       : [S, 2*D]         contiguous
    k_body_ptr,     # k_body     : non-contiguous; stride Sp1*D per head
    reshape_k_ptr,  # reshape_k  : [1, H,  S,  D]  contiguous
    k_out_ptr,      # k output   : [1, H, Sp1, D]  contiguous
    H, S, Sp1,
    D: tl.constexpr, D2: tl.constexpr,
):
    pid    = tl.program_id(0)      # in [0, H * Sp1)
    h      = pid // Sp1
    s_out  = pid % Sp1
    is_cls = (s_out == 0)
    s      = tl.where(is_cls, 0, s_out - 1)   # safe index into body arrays
    d      = tl.arange(0, D)
    out_off = (h * Sp1 + s_out) * D

    # ── Q ──────────────────────────────────────────────────────────────────
    q_cls_val = tl.load(q_cls_ptr + h * D + d)
    body_off  = (h * S + s) * D
    q_v       = tl.load(q_ptr        + body_off + d)
    rq_v      = tl.load(reshape_q_ptr + body_off + d)
    cos_v     = tl.load(cos_ptr + s * D + d)
    sin_v     = tl.load(sin_ptr + s * D + d)
    q_body    = q_v * cos_v + rq_v * sin_v
    tl.store(q_out_ptr + out_off + d, tl.where(is_cls, q_cls_val, q_body))

    # ── K ──────────────────────────────────────────────────────────────────
    k_cls_val = tl.load(k_ptr + h * Sp1 * D + d)   # cls token row (s_out=0)
    kb_off    = h * Sp1 * D + s * D                  # k_body stride=Sp1*D per head
    kb_v      = tl.load(k_body_ptr  + kb_off + d)
    rk_off    = (h * S + s) * D
    rk_v      = tl.load(reshape_k_ptr + rk_off + d)
    i0_off    = s * D2
    ck        = tl.load(in0_ptr + i0_off + d)        # cos_k[s]
    sk        = tl.load(in0_ptr + i0_off + D + d)    # sin_k[s]
    k_body    = kb_v * sk + rk_v * ck
    tl.store(k_out_ptr + out_off + d, tl.where(is_cls, k_cls_val, k_body))


# ── Pattern: shape-agnostic ──────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, reshape_q, k_body, reshape_k, in_6):
    # ── Q path ──
    tmp_1  = in_3 * in_1
    tmp_7  = reshape_q * in_5
    tmp_8  = tmp_1 + tmp_7
    tmp_9  = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)

    # ── K path (k_body external to avoid leak to tmp_17/tmp_19) ──
    tmp_11 = in_4[slice(None, None, None), slice(None, None, None),
                  slice(None, 1, None),    slice(None, None, None)]
    tensor_split = in_0.tensor_split(2, -1)
    cos_k  = tensor_split[0]
    sin_k  = tensor_split[1]
    tmp_16 = k_body * sin_k
    tmp_22 = reshape_k * cos_k
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(in_6)

    return (tmp_25, tmp_10)


# ── replacement_args ─────────────────────────────────────────────────────────
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, reshape_q, k_body, reshape_k, in_6):
    return (in_2, in_3, in_1, reshape_q, in_5, in_4, in_0, k_body, reshape_k, in_6)


# ── Fused implementation ─────────────────────────────────────────────────────
@torch.fx.wrap
def _rope_fused_universal_impl(in_2, in_3, in_1, reshape_q, in_5,
                                in_4, in_0, k_body, reshape_k, in_6):
    H   = in_3.shape[1]
    S   = in_3.shape[2]
    Sp1 = S + 1

    # Single combined allocation for both Q and K outputs (one cudaMalloc)
    combined = torch.empty((2, H, Sp1, 64), dtype=in_3.dtype, device=in_3.device)
    q_out = combined[0:1]   # [1, H, Sp1, 64]
    k_out = combined[1:2]   # [1, H, Sp1, 64]

    _rope_tail_paired[(H * Sp1,)](
        in_2, in_3, in_1, reshape_q, in_5, q_out,
        in_4, in_0, k_body, reshape_k, k_out,
        H, S, Sp1, D=64, D2=128,
    )

    return k_out.to(in_6.dtype), q_out.to(in_6.dtype)


def _rope_fused_universal(in_2, in_3, in_1, reshape_q, in_5,
                           in_4, in_0, k_body, reshape_k, in_6):
    result = _rope_fused_universal_impl(
        in_2, in_3, in_1, reshape_q, in_5,
        in_4, in_0, k_body, reshape_k, in_6)
    return result[0], result[1]


def replacement_func():
    return _rope_fused_universal