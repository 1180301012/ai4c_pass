import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: relu → 3× identical max_pool2d → cat along channel dim
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


# ──────────────────────────────────────────────────────────────────────────────
# Fully-fused Triton kernel:
#   • Applies ReLU inline (reads from original input, no in-place mutation)
#   • Computes 5×5 max-pool once (3× identical pool calls collapsed)
#   • Writes directly into the 4-slot output tensor
#
# Grid: (ceil(H*W / BLOCK_HW), C, B)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=16),
    ],
    key=['H', 'W'],
)
@triton.jit
def _fused_relu_maxpool_cat_kernel(
    src_ptr,   # [B, C, H, W]  – original input
    out_ptr,   # [B, 4*C, H, W]
    C, H, W,
    CHW,       # C * H * W
    BLOCK_HW: tl.constexpr,
):
    """
    Each thread block owns BLOCK_HW output spatial positions for a fixed (batch, channel).

    For each position:
      • relu_val  = max(input[b, c, oh, ow], 0)     → written to out slot 0
      • pool_val  = max over 5×5 window of relu(input) → written to out slots 1, 2, 3
    """
    pid_hw = tl.program_id(0)   # spatial block
    pid_c  = tl.program_id(1)   # channel
    pid_b  = tl.program_id(2)   # batch

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < H * W

    # Decompose flat hw → (oh, ow)
    oh = hw_offs // W
    ow = hw_offs % W

    src_base = pid_b * CHW + pid_c * H * W   # base offset into src for this (b, c)

    # ── Load & ReLU the center pixel (always in-bounds) ──────────────────────
    relu_center = tl.load(src_ptr + src_base + oh * W + ow, mask=mask_hw, other=0.0)
    relu_center = tl.maximum(relu_center, 0.0)

    # ── Compute 5×5 max-pool with inline ReLU ────────────────────────────────
    # Initialise max with center (correct dtype, guaranteed in-bounds)
    max_val = relu_center

    for kh in range(5):
        for kw in range(5):
            ih = oh + (kh - 2)
            iw = ow + (kw - 2)
            in_bounds = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            val = tl.load(src_ptr + src_base + ih * W + iw,
                          mask=in_bounds, other=0.0)
            val = tl.maximum(val, 0.0)          # inline ReLU
            max_val = tl.maximum(max_val, val)

    # ── Write output ─────────────────────────────────────────────────────────
    # Slot 0: relu'd input
    s0 = pid_b * 4 * CHW + pid_c * H * W
    tl.store(out_ptr + s0 + hw_offs, relu_center, mask=mask_hw)

    # Slots 1, 2, 3: max-pool result (all three are identical)
    s1 = pid_b * 4 * CHW + 1 * CHW + pid_c * H * W
    s2 = pid_b * 4 * CHW + 2 * CHW + pid_c * H * W
    s3 = pid_b * 4 * CHW + 3 * CHW + pid_c * H * W
    tl.store(out_ptr + s1 + hw_offs, max_val, mask=mask_hw)
    tl.store(out_ptr + s2 + hw_offs, max_val, mask=mask_hw)
    tl.store(out_ptr + s3 + hw_offs, max_val, mask=mask_hw)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel wrapper  (must be decorated with @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_relu_maxpool_cat(x):
    """
    Single-kernel replacement for:
        relu(x)  +  max_pool2d × 3  +  cat([relu, pool, pool, pool], dim=1)

    Speedups:
      • Pool computed only ONCE (3× reduction in pool work)
      • ReLU fused into the pool kernel (no separate pass)
      • Cat fused into the same kernel (no extra memory allocation for intermediates)
    """
    B, C, H, W = x.shape
    CHW = C * H * W
    out = torch.empty(B, 4 * C, H, W, dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(H * W, meta['BLOCK_HW']), C, B)
    _fused_relu_maxpool_cat_kernel[grid](x, out, C, H, W, CHW)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Required hook: return the wrapper function (do NOT call it)
# ──────────────────────────────────────────────────────────────────────────────

def replacement_func():
    return fused_relu_maxpool_cat