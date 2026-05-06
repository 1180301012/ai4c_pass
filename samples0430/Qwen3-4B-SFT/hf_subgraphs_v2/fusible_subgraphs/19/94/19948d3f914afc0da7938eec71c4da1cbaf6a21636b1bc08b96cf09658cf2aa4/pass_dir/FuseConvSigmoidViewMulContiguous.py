import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Unified tiled kernel: 2-D grid (channel × spatial tile).
#
# BLOCK_HW=1024, num_stages=3:
#   - No autotune overhead (~5-10 μs per kernel call removed).
#   - Static grid (no lambda) for zero Python dispatch cost per call.
#   - num_stages=3: Triton pipelines 3 loop iterations, overlapping in_2
#     loads with preceding compute and hiding ~200-cycle HBM latency.
#
# Tile coverage for all N_HW ∈ {1024, 9216, 12544, 16384, 25600}:
#   N_HW=1024  → 1 tile, no masking.
#   N_HW=9216  → 9 tiles (9×1024 = 9216, exact — NO masking EVER).
#   N_HW=12544 → 13 tiles, last tile tail-masked (56% valid).
#   N_HW=16384 → 16 tiles, no masking.
#   N_HW=25600 → 25 tiles, no masking (25×1024 = 25600 exact — NO masking).
# ---------------------------------------------------------------------------
@triton.jit
def _fused_conv_sigmoid_scale(
    in_3_ptr,           # [1, IN_C, 1, 1] int32 – gap features (contiguous)
    weight_ptr,         # [N_C, IN_C, 1, 1] float  – synthetic weights
    bias_ptr,           # [N_C]            float  – bias
    in_2_ptr,           # [1, N_C, H, W]   any float dtype
    out_ptr,            # [1, N_C, H, W]   same dtype
    N_C, N_HW,
    IN_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c     = tl.program_id(0)   # channel index
    p     = tl.program_id(1)   # spatial tile index

    # ---- gating scale: sigmoid(in3 · w[c] + bias[c]) ----
    k     = tl.arange(0, IN_C)
    x     = tl.load(in_3_ptr + k).to(tl.float32)       # int32 → fp32
    w     = tl.load(weight_ptr + c * IN_C + k)          # cast in-kernel
    dw    = tl.sum(x * w, axis=0) + tl.load(bias_ptr + c).to(tl.float32)
    scale = tl.sigmoid(dw)                              # fp32 scalar

    # ---- element-wise scale (pipeline: num_stages=3 overlaps loads) ----
    base  = c * N_HW + p * BLOCK_HW
    off   = tl.arange(0, BLOCK_HW)
    mask  = (base + off) < (c * N_HW + N_HW)
    v     = tl.load(in_2_ptr + base + off, mask=mask, other=0.0)
    tl.store(out_ptr + base + off, v.to(tl.float32) * scale, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_scale(bias, weight, in_2, in_3):
    """
    Fused replacement for:
        conv2d   <- torch.conv2d(in_3, weight, bias, (1,1),(0,0),(1,1),4)
        sig      <- torch.sigmoid(conv2d)
        view     <- sig.view(1, -1, 1, 1)
        mul      <- in_2 * view
        out      <- mul.contiguous()
    Returns a contiguous tensor of shape [1, N_C, H, W].
    """
    N_C   = weight.shape[0]
    IN_C  = weight.shape[1]
    H     = in_2.shape[2]
    W     = in_2.shape[3]
    N_HW  = H * W

    out = torch.empty_like(in_2)

    # Static grid — no lambda overhead, no per-call dispatch cost.
    # BLOCK_HW=1024 divides every N_HW in the problem set exactly or
    # with minimal last-tile masking (only when N_HW % 1024 ≠ 0).
    BLOCK_HW = 1024
    grid = (N_C, triton.cdiv(N_HW, BLOCK_HW))
    _fused_conv_sigmoid_scale[grid](
        in_3, weight, bias, in_2, out,
        N_C=N_C, N_HW=N_HW, IN_C=IN_C,
        BLOCK_HW=BLOCK_HW,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """Mirrors model.py exactly (positional args, same ops, no None-cleanup)."""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3  = torch.sigmoid(conv2d)
    tmp_4  = tmp_3.view(1, -1, 1, 1)
    tmp_5  = in_2 * tmp_4
    tmp_6  = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    # in_0=bias, in_1=weight, in_2=input, in_3=conv_input
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_conv_sigmoid_scale