import torch
import triton
import triton.language as tl
from torch import inf


# ─── Pattern ────────────────────────────────────────────────────────────────
def pattern(pow_out, in_5, in_4, in_2):
    """
    Match the GAE/RECT_L edge-weight normalisation subgraph.

    Making pow_out an ARGUMENT avoids NOT_CONTAINED because tmp_2
    (the pow result) "leaks" to eq(inf)/masked_fill_ nodes outside
    the matched subgraph.  As an external placeholder, pow_out's
    external users don't violate the containment check.
    """
    tmp_5 = pow_out[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = pow_out[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


# ─── Replacement args ───────────────────────────────────────────────────────
def replacement_args(pow_out, in_5, in_4, in_2):
    return (pow_out, in_5, in_4, in_2)


# ─── Triton kernel ──────────────────────────────────────────────────────────
@triton.jit
def _edge_weight_kernel(
    pow_ptr,
    in5_ptr, in4_ptr, in2_ptr,
    out_ptr,
    E,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: out[e] = pow_out[row[e]] * w[e] * pow_out[col[e]]
    Scatter-loads from pow_ptr; after warmup the small pow array
    (<= 2 KB) stays in L1 cache.
    """
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    e_mask = e_offs < E

    w = tl.load(in4_ptr + e_offs, mask=e_mask, other=0.0)
    row_i = tl.load(in5_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    col_i = tl.load(in2_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)

    scale_row = tl.load(pow_ptr + row_i, mask=e_mask, other=0.0)
    scale_col = tl.load(pow_ptr + col_i, mask=e_mask, other=0.0)

    result = scale_row * w.to(tl.float32) * scale_col
    tl.store(out_ptr + e_offs, result, mask=e_mask)


# ─── Kernel wrapper ─────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_power_weight(pow_out, in_5, in_4, in_2):
    dev = pow_out.device
    E = in_5.shape[0]
    out = torch.empty(E, dtype=in_4.dtype, device=dev)
    # Fixed BLOCK_SIZE=128 → 4 warps per block (sweet spot for these graphs)
    #   E=1100: 9 blocks → ~36 warps total on 28 SMs
    #   E=256:  2 blocks → ~8 warps total
    BLOCK_SIZE = 128
    grid = ((E + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _edge_weight_kernel[grid](
        pow_ptr=pow_out,
        in5_ptr=in_5,
        in4_ptr=in_4,
        in2_ptr=in_2,
        out_ptr=out,
        E=E,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ─── Replacement func ───────────────────────────────────────────────────────
def replacement_func():
    return fused_power_weight