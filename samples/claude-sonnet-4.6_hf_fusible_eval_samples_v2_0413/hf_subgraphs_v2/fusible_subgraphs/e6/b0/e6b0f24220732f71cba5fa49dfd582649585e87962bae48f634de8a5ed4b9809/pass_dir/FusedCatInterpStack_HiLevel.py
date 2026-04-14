"""
Fused pass: cat + interp + interp + stack  — HIGH-LEVEL OP VARIANT
Uses torch.fx.wrap to keep F.interpolate as a single FX node (leaf),
then matches the torch-level graph (torch.cat / F.interpolate / torch.stack).
"""

import torch
import triton
import triton.language as tl

# Register F.interpolate as an FX leaf so symbolic_trace does NOT
# trace through its Python control-flow.  This is required to produce a
# single-node pattern that matches the GraphModule's stored graph.
torch.fx.wrap('torch.nn.functional.interpolate')


# ---------------------------------------------------------------------------
# Pattern  — mirrors model.py exactly at the torch-functional level
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel  (identical to FusedCatInterpStack_40x40.py)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['slice_size'],
)
@triton.jit
def _hl_fused_kernel(
    out_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    slice_size,
    total_elements,
    CHW:        tl.constexpr,
    HW:         tl.constexpr,
    CHsWs:      tl.constexpr,
    HsWs:       tl.constexpr,
    Ws:         tl.constexpr,
    W:          tl.constexpr,
    C_half:     tl.constexpr,
    C_half_HW:  tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < total_elements

    i     = offsets // slice_size
    local = offsets - i * slice_size

    b    = local // CHW
    rem  = local - b * CHW
    c    = rem   // HW
    rem2 = rem   - c * HW
    h    = rem2  // W
    w    = rem2  - h * W

    src0 = local
    src1 = b * CHsWs + c * HsWs + (h >> 1) * Ws + (w >> 1)

    c_for_in3 = tl.where(c >= C_half, c - C_half, c)
    src2_a = b * C_half_HW + c         * HW + rem2
    src2_b = b * C_half_HW + c_for_in3 * HW + rem2

    m0  = mask & (i == 0)
    m1  = mask & (i == 1)
    m2a = mask & (i == 2) & (c <  C_half)
    m2b = mask & (i == 2) & (c >= C_half)

    v0  = tl.load(in0_ptr + src0,   mask=m0,  other=0.0)
    v1  = tl.load(in1_ptr + src1,   mask=m1,  other=0.0)
    v2a = tl.load(in2_ptr + src2_a, mask=m2a, other=0.0)
    v2b = tl.load(in3_ptr + src2_b, mask=m2b, other=0.0)

    val = tl.where(i == 0, v0,
          tl.where(i == 1, v1,
          tl.where(c < C_half, v2a, v2b)))

    tl.store(out_ptr + offsets, val, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _hl_fused_cat_interp_stack(in_0, in_1, in_2, in_3):
    B       = in_0.shape[0]
    C       = 512
    H       = 40
    W_dim   = 40
    Hs      = 20
    Ws      = 20
    C_half  = 256

    CHW       = C     * H    * W_dim
    HW        = H     * W_dim
    CHsWs     = C     * Hs   * Ws
    HsWs      = Hs    * Ws
    C_half_HW = C_half * H   * W_dim

    slice_size     = B * CHW
    total_elements = 3 * slice_size

    out = torch.empty((3, B, C, H, W_dim), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (
        (total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )

    _hl_fused_kernel[grid](
        out, in_0, in_1, in_2, in_3,
        slice_size, total_elements,
        CHW, HW, CHsWs, HsWs, Ws, W_dim,
        C_half, C_half_HW,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement hook
# ---------------------------------------------------------------------------

def replacement_func():
    return _hl_fused_cat_interp_stack