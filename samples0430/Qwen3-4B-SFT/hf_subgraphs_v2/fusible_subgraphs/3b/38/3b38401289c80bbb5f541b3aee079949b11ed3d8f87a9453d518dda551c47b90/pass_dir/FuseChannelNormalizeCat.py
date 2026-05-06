import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ──────────────────────────────────────────────────────────────────────────────
# Design:
#  • 2-D grid (B, ceil(HW / BLOCK_SIZE)) so even small-B cases get more CTAs.
#  • BLOCK_SIZE is chosen heuristically from HW to avoid pathological small-block
#    regimes (tiny worked-per-CTA that trigger register spilling due to L1 cache
#    pressure).
#  • The three independent load streams are live at once → GPU can pipeline them.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_channel_normalize_2d_kernel(
    in0_ptr,    # [B, C, H, W] contiguous bfloat16 / float16
    in1_ptr,    # [B, 1, H, W] contiguous bfloat16 / float16
    out_ptr,    # [B, 3, H, W] contiguous bfloat16 / float16
    B,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    b_idx        = tl.program_id(0)      # batch
    hw_chunk_idx = tl.program_id(1)      # spatial chunk

    hw_start  = hw_chunk_idx * BLOCK_SIZE
    hw_ids    = hw_start + tl.arange(0, BLOCK_SIZE)
    mask      = hw_ids < HW

    out_b_base = b_idx * (3 * HW)
    in0_b_base = b_idx * C * HW

    # Load all three channel sources simultaneously – GPU pipelines these.
    a   = tl.load(in1_ptr   + b_idx * HW  + hw_ids, mask=mask, other=0.0)
    b_v = tl.load(in0_ptr   + in0_b_base + HW  + hw_ids, mask=mask, other=0.0)
    c_v = tl.load(in0_ptr   + in0_b_base + 2*HW + hw_ids, mask=mask, other=0.0)

    # Fused affine transforms, computed in float32 for numerical precision.
    a_f  = a.to(tl.float32) * 0.458 - 0.030000000000000027
    b_f  = b_v.to(tl.float32) * 0.448 - 0.08799999999999997
    c_f  = c_v.to(tl.float32) * 0.45  - 0.18799999999999994

    tl.store(out_ptr + out_b_base + 0*HW + hw_ids, a_f.to(a.dtype), mask=mask)
    tl.store(out_ptr + out_b_base + 1*HW + hw_ids, b_f.to(a.dtype), mask=mask)
    tl.store(out_ptr + out_b_base + 2*HW + hw_ids, c_f.to(a.dtype), mask=mask)


@torch.fx.wrap
def fused_channel_normalize(in_0, in_1):
    B, C, H, W = in_0.shape
    HW  = H * W
    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    # Choose BLOCK_SIZE to give ≥4 elements per thread, small enough for good
    # occupancy on tiny tensors, large enough to stay within L1 cache.
    #
    #   HW ≤  512  → BLOCK_SIZE=512  (2 hw_chunks when HW=512,  B programs)
    #  512 < HW ≤ 2048 → BLOCK_SIZE=1024
    #  2048 < HW  ≤ 8192 → BLOCK_SIZE=2048
    #  8192 < HW               → BLOCK_SIZE=4096
    #
    # These rules keep the number of CTAs in a healthy range.
    if HW <= 512:
        BLOCK_SIZE = 512
    elif HW <= 2048:
        BLOCK_SIZE = 1024
    elif HW <= 8192:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096

    num_hw_chunks = (HW + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B, num_hw_chunks)

    _fused_channel_normalize_2d_kernel[grid](
        in_0, in_1, out,
        B, C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_channel_normalize