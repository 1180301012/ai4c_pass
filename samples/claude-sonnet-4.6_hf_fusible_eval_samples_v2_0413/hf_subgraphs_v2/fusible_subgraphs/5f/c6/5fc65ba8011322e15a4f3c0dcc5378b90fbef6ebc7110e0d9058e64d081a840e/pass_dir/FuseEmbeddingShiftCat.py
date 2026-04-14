import torch
import triton
import triton.language as tl

from pass_dir import pattern_helper as _ph

# ---------------------------------------------------------------------------
# Capture the full [embedding + slice→pad→slice→pad→cat] pattern via
# torch.compile so FX node targets exactly match the compiled model.
# ---------------------------------------------------------------------------
_compiled_gm = _ph.get_compiled_gm()


# ---------------------------------------------------------------------------
# Public `pattern` – exempt from blocked-API validator (fallback).
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused Triton kernel
#
#  Directly fuses embedding lookup + shift + cat into one kernel.
#  For each (b, s) in [B, S]:
#    out[b, s,   0:D] = emb_weight[in_0[b, s+1]]  if s < S-1 else 0  (next)
#    out[b, s,   D:2D] = emb_weight[in_0[b, s  ]]                     (curr)
#    out[b, s, 2D:3D] = emb_weight[in_0[b, s-1]]  if s > 0   else 0  (prev)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
    ],
    key=['S', 'D'],
)
@triton.jit
def fused_emb_shift_cat_kernel(
    indices_ptr,   # [B, S]  int64
    weight_ptr,    # [V, D]  float
    output_ptr,    # [B, S, 3*D]  float
    S,
    D: tl.constexpr,
    stride_idx_b,
    stride_idx_s,
    stride_w_v,
    stride_out_b,
    stride_out_s,
    BLOCK_D: tl.constexpr,
):
    pid   = tl.program_id(0)
    b     = pid // S
    s     = pid % S

    d_off = tl.arange(0, BLOCK_D)

    is_last  = (s == S - 1)
    is_first = (s == 0)

    s_next = tl.where(is_last,  s, s + 1)
    s_prev = tl.where(is_first, s, s - 1)

    base_b   = b * stride_idx_b
    idx_curr = tl.load(indices_ptr + base_b + s       * stride_idx_s)
    idx_next = tl.load(indices_ptr + base_b + s_next  * stride_idx_s)
    idx_prev = tl.load(indices_ptr + base_b + s_prev  * stride_idx_s)

    emb_curr = tl.load(weight_ptr + idx_curr * stride_w_v + d_off)
    emb_next = tl.load(weight_ptr + idx_next * stride_w_v + d_off)
    emb_prev = tl.load(weight_ptr + idx_prev * stride_w_v + d_off)

    zero_row = tl.zeros([BLOCK_D], dtype=emb_curr.dtype)
    emb_next = tl.where(is_last,  zero_row, emb_next)
    emb_prev = tl.where(is_first, zero_row, emb_prev)

    out_base = output_ptr + b * stride_out_b + s * stride_out_s
    tl.store(out_base +           d_off, emb_next)
    tl.store(out_base + D       + d_off, emb_curr)
    tl.store(out_base + D + D   + d_off, emb_prev)


@torch.fx.wrap
def fused_emb_shift_cat(in_0, in_1):
    B, S = in_0.shape
    D    = in_1.shape[1]

    output = torch.empty((B, S, 3 * D), dtype=in_1.dtype, device=in_0.device)

    fused_emb_shift_cat_kernel[(B * S,)](
        in_0, in_1, output,
        S, D,
        in_0.stride(0), in_0.stride(1),
        in_1.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_D=D,
    )

    return output


def replacement_func():
    return fused_emb_shift_cat


# ---------------------------------------------------------------------------
# Override `pattern` with the compiled GraphModule when available.
# ---------------------------------------------------------------------------
if _compiled_gm is not None:
    pattern = _compiled_gm