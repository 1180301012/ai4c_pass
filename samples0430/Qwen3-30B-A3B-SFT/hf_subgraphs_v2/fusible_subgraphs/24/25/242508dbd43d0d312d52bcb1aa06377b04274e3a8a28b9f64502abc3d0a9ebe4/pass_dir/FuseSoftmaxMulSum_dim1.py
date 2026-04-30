import torch
import triton
import triton.language as tl


# ── K=2 fused softmax-weighted-sum (B=1) ────────────────────────────────────
# out[c,h,w] = w0*x0[c,h,w] + w1*x1[c,h,w]
# w1 = exp(v1-max)/(exp(v0-max)+exp(v1-max))
# out = x0 + (x1-x0)*w1
#
# BLOCK_HW adapts to HW:
#   HW ≤ 256 → BLOCK_HW=256 (1 spatial block/channel, no loop)
#   HW > 256 → BLOCK_HW=64  (4-13 blocks/channel, 100% warp occupancy)
# Grid is (ceil(HW/BLOCK_HW), C) for all cases.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def softmax_mul_sum_kernel(
    in0_ptr,              # [1, 2, C, H, W]
    in1_ptr,              # [1, 2, C]
    out_ptr,              # [1, C, H, W]
    HW,                   # H * W
    C,                    # channels
    CHW,                  # C * H * W
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)   # spatial tile
    c      = tl.program_id(1)   # channel index

    # ── per-channel K=2 softmax ───────────────────────────────────────────────
    v0 = tl.load(in1_ptr + c).to(tl.float32)
    v1 = tl.load(in1_ptr + C + c).to(tl.float32)

    vmax = tl.maximum(v0, v1)
    e0   = tl.exp(v0 - vmax)
    e1   = tl.exp(v1 - vmax)
    w    = e1 / (e0 + e1)

    # ── spatial tile ──────────────────────────────────────────────────────────
    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW
    base   = c * HW

    x0 = tl.load(in0_ptr + base + hw_off,        mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in0_ptr + base + CHW + hw_off,  mask=mask, other=0.0).to(tl.float32)

    result = x0 + (x1 - x0) * w

    tl.store(out_ptr + base + hw_off, result, mask=mask)


@torch.fx.wrap
def triton_softmax_mul_sum(in_0, in_1):
    B, K, C, H, W = in_0.shape
    HW  = H * W
    CHW = C * HW

    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    # Adaptive BLOCK_HW:
    #   HW ≤ 256 → 256 (1 block/channel)
    #     num_warps=4 → 4×4=16 blocks/SM × 4 warps = 64 warps = 100% occupancy
    #   HW > 256 → 64 (4-13 blocks/channel)
    #     num_warps=2 → 32 blocks/SM  × 2 warps = 64 warps = 100% occupancy
    if HW <= 256:
        BLOCK_HW = 256
        NW = 4
    else:
        BLOCK_HW = 64
        NW = 2
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (num_hw_blocks, B * C)

    softmax_mul_sum_kernel[grid](
        in_0, in_1, out,
        HW, C, CHW,
        BLOCK_HW=BLOCK_HW,
        num_warps=NW,
    )

    return out


# ── Pattern / replacement API ─────────────────────────────────────────────────

def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_softmax_mul_sum