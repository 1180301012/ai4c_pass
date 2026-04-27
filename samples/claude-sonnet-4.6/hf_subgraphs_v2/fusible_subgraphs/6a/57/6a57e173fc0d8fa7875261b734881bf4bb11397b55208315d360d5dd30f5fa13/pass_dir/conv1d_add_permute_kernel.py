"""
Shared Triton kernels.

Kernel 1: grouped_conv1d_kernel
  Grouped 1-D depthwise conv.
  Input / output: [B, G, H, W].  Weight: [G, 1, KSIZE, 1].

Kernel 2: permute_0213_contiguous_kernel
  Transpose [B, G, H, W] -> [B, H, G, W] (== permute(0,2,1,3) + contiguous).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: grouped 1-D conv  (no in-place add, no permute)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 8},  num_warps=1),
        triton.Config({"BLOCK_W": 16}, num_warps=2),
        triton.Config({"BLOCK_W": 32}, num_warps=4),
        triton.Config({"BLOCK_W": 64}, num_warps=4),
        triton.Config({"BLOCK_W": 128}, num_warps=8),
    ],
    key=["B", "G", "H", "W"],
)
@triton.jit
def grouped_conv1d_kernel(
    in_ptr, weight_ptr, out_ptr,
    B, G, H, W,
    KSIZE: tl.constexpr,   # 65
    PAD:   tl.constexpr,   # 32
    BLOCK_W: tl.constexpr,
):
    pid  = tl.program_id(0)
    h    = pid % H
    tmp  = pid // H
    g    = tmp % G
    b    = tmp // G

    w_base     = weight_ptr + g * KSIZE
    in_bg_base = in_ptr + (b * G + g) * (H * W)

    w_offs = tl.arange(0, BLOCK_W)
    mask_w = w_offs < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for k in range(KSIZE):
        h_in    = h + k - PAD
        valid   = (h_in >= 0) & (h_in < H)
        h_safe  = tl.where(valid, h_in, 0)
        valid_f = tl.where(valid, 1.0, 0.0).to(tl.float32)

        in_val = tl.load(in_bg_base + h_safe * W + w_offs, mask=mask_w, other=0.0)
        wk     = tl.load(w_base + k)
        acc   += valid_f * wk.to(tl.float32) * in_val.to(tl.float32)

    out_base = out_ptr + (b * G + g) * (H * W) + h * W
    tl.store(out_base + w_offs, acc.to(in_ptr.dtype.element_ty), mask=mask_w)


# ---------------------------------------------------------------------------
# Kernel 2: permute(0,2,1,3) + contiguous : [B,G,H,W] -> [B,H,G,W]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 8},  num_warps=1),
        triton.Config({"BLOCK_W": 16}, num_warps=2),
        triton.Config({"BLOCK_W": 32}, num_warps=4),
        triton.Config({"BLOCK_W": 64}, num_warps=4),
        triton.Config({"BLOCK_W": 128}, num_warps=8),
    ],
    key=["B", "G", "H", "W"],
)
@triton.jit
def permute_0213_contiguous_kernel(
    in_ptr, out_ptr,
    B, G, H, W,
    BLOCK_W: tl.constexpr,
):
    pid  = tl.program_id(0)
    h    = pid % H
    tmp  = pid // H
    g    = tmp % G
    b    = tmp // G

    w_offs = tl.arange(0, BLOCK_W)
    mask_w = w_offs < W

    in_base  = in_ptr  + (b * G + g) * (H * W) + h * W
    out_base = out_ptr + (b * H + h) * (G * W) + g * W

    vals = tl.load(in_base + w_offs, mask=mask_w)
    tl.store(out_base + w_offs, vals, mask=mask_w)


# ---------------------------------------------------------------------------
# Kernel 3: FULL FUSION – conv1d + in-place-add + permute(0,2,1,3)
# Input layout:   in1, in2  shape [B, G, H, W]
# Weight layout:  [G, 1, KSIZE, 1]  ->  strides [KSIZE, KSIZE, 1, 1]
# Output layout:  [B, H, G, W]  (permuted, contiguous)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 8},  num_warps=2),
        triton.Config({"BLOCK_W": 16}, num_warps=4),
        triton.Config({"BLOCK_W": 32}, num_warps=4),
        triton.Config({"BLOCK_W": 64}, num_warps=8),
    ],
    key=["B", "G", "H", "W"],
)
@triton.jit
def fused_conv1d_add_permute_kernel(
    in1_ptr, in2_ptr, weight_ptr, out_ptr,
    B, G, H, W,
    KSIZE: tl.constexpr,   # 65
    PAD:   tl.constexpr,   # 32
    BLOCK_W: tl.constexpr,
):
    pid  = tl.program_id(0)
    h    = pid % H
    tmp  = pid // H
    g    = tmp % G
    b    = tmp // G

    w_base      = weight_ptr + g * KSIZE
    in2_bg_base = in2_ptr + (b * G + g) * (H * W)

    w_offs = tl.arange(0, BLOCK_W)
    mask_w = w_offs < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for k in range(KSIZE):
        h_in    = h + k - PAD
        valid   = (h_in >= 0) & (h_in < H)
        h_safe  = tl.where(valid, h_in, 0)
        valid_f = tl.where(valid, 1.0, 0.0).to(tl.float32)

        in2_val = tl.load(in2_bg_base + h_safe * W + w_offs, mask=mask_w, other=0.0)
        wk      = tl.load(w_base + k)
        acc    += valid_f * wk.to(tl.float32) * in2_val.to(tl.float32)

    in1_base = in1_ptr + (b * G + g) * (H * W) + h * W
    in1_val  = tl.load(in1_base + w_offs, mask=mask_w)
    result   = (in1_val.to(tl.float32) + acc).to(in1_ptr.dtype.element_ty)

    out_base = out_ptr + (b * H + h) * (G * W) + g * W
    tl.store(out_base + w_offs, result, mask=mask_w)