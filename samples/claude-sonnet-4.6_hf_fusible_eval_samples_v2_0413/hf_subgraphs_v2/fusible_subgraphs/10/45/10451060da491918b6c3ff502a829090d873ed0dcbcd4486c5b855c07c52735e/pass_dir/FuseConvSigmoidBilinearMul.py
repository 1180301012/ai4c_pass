import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: conv2d -> sigmoid -> bilinear_interpolate -> multiply
    Uses ATen-level ops matching the _decomposed graph representation.
    """
    conv2d = torch.ops.aten.convolution.default(in_1, in_0, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    tmp_2 = torch.ops.aten.sigmoid.default(conv2d)
    tmp_3 = torch.ops.aten.upsample_bilinear2d.vec(tmp_2, [64, 128], False, None)
    tmp_4 = torch.ops.aten.mul.Tensor(in_2, tmp_3)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fuse conv2d + sigmoid + bilinear-upsample + elementwise mul
#
# Shapes:
#   in_0 : [128, 960, 1, 1]  -- conv weight,  treated as [N_OC=128, N_IC=960]
#   in_1 : [  1, 960, 1, 4]  -- conv input,   treated as [N_IC=960,  IN_W=4]
#   in_2 : [  1, 128, 64, 128] -- scale tensor, treated as [N_OC=128, HW=8192]
#   out  : [  1, 128, 64, 128]
#
# Strategy:
#   - Grid: (N_OC=128,  ceil(HW / BLOCK_HW))
#   - Each program handles one output channel c and a spatial tile of BLOCK_HW elements.
#   - Step 1: compute conv[c, w_in] = dot(in_0[c,:], in_1[:,w_in]) for w_in in 0..3
#             (only 960-element dot product -- tiny, cached in L2 across hw-tiles)
#   - Step 2: sigmoid -> 4 scalar values s0..s3
#   - Step 3: for each output (h_out, w_out) in tile:
#             * bilinear interp from [s0..s3] using w_out only
#               (H_in=1 so no H interpolation needed)
#             * multiply with in_2[c, h_out*OUT_W+w_out]
#             * store
#
# Memory savings vs baseline:
#   Baseline: write 2 MB upsampled tensor, then read it back for multiply → ~4 MB extra
#   Fused:    read in_2 once, write output once → ~4 MB total (half the traffic)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16),
    ],
    key=['N_OC', 'N_IC', 'OUT_HW'],
)
@triton.jit
def _fused_conv_sig_bilinear_mul_kernel(
    in_0_ptr,          # [N_OC, N_IC]  float16/bf16
    in_1_ptr,          # [N_IC, IN_W]  float16/bf16
    in_2_ptr,          # [N_OC, OUT_HW] float16/bf16
    out_ptr,           # [N_OC, OUT_HW] float16/bf16  (output)
    N_OC:    tl.constexpr,   # 128
    N_IC:    tl.constexpr,   # 960
    IN_W:    tl.constexpr,   # 4
    OUT_H:   tl.constexpr,   # 64
    OUT_W:   tl.constexpr,   # 128
    OUT_HW:  tl.constexpr,   # OUT_H * OUT_W = 8192
    N_IC_PAD: tl.constexpr,  # 1024  (next power-of-2 >= N_IC, for tl.arange)
    IS_BF16: tl.constexpr,   # bool
    BLOCK_HW: tl.constexpr,  # spatial tile size (autotuned)
):
    c      = tl.program_id(0)   # output channel
    hw_pid = tl.program_id(1)   # spatial tile id

    # ------------------------------------------------------------------
    # Step 1: conv2d for channel c across all IN_W=4 input-x positions
    #   conv[c, w] = sum_ic( in_0[c,ic] * in_1[ic,w] )
    # ------------------------------------------------------------------
    ic_offs = tl.arange(0, N_IC_PAD)          # [N_IC_PAD]
    ic_mask = ic_offs < N_IC

    # Load in_0[c, :] – weight vector for output channel c
    w0 = tl.load(in_0_ptr + c * N_IC + ic_offs,
                 mask=ic_mask, other=0.0).to(tl.float32)   # [N_IC_PAD]

    # Load each column of in_1 (stride IN_W in memory) and dot with w0
    # in_1[ic, w] is at offset  ic*IN_W + w
    col0 = tl.load(in_1_ptr + ic_offs * IN_W + 0, mask=ic_mask, other=0.0).to(tl.float32)
    col1 = tl.load(in_1_ptr + ic_offs * IN_W + 1, mask=ic_mask, other=0.0).to(tl.float32)
    col2 = tl.load(in_1_ptr + ic_offs * IN_W + 2, mask=ic_mask, other=0.0).to(tl.float32)
    col3 = tl.load(in_1_ptr + ic_offs * IN_W + 3, mask=ic_mask, other=0.0).to(tl.float32)

    cv0 = tl.sum(w0 * col0, axis=0)   # scalar
    cv1 = tl.sum(w0 * col1, axis=0)
    cv2 = tl.sum(w0 * col2, axis=0)
    cv3 = tl.sum(w0 * col3, axis=0)

    # ------------------------------------------------------------------
    # Step 2: sigmoid – 4 scalars
    # ------------------------------------------------------------------
    s0 = tl.sigmoid(cv0)
    s1 = tl.sigmoid(cv1)
    s2 = tl.sigmoid(cv2)
    s3 = tl.sigmoid(cv3)

    # ------------------------------------------------------------------
    # Step 3: bilinear interp (H trivial since H_in=1) + multiply + store
    # ------------------------------------------------------------------
    hw_base = hw_pid * BLOCK_HW
    hw_offs = hw_base + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
    hw_mask = hw_offs < OUT_HW

    # Output column index (row doesn't affect bilinear since H_in=1)
    w_out = hw_offs % OUT_W    # [0, OUT_W)

    # Source x coordinate for bilinear (align_corners=False):
    #   src_x = (w_out + 0.5) * (IN_W / OUT_W) - 0.5
    #         = (w_out + 0.5) * 0.03125 - 0.5
    src_x = (w_out.to(tl.float32) + 0.5) * (4.0 / 128.0) - 0.5

    # Floor of src_x  (src_x ∈ [-0.485, 3.485])
    # For non-negative values: floor = int-truncation-toward-zero  ✓
    # For negative values (small region near w_out=0):
    #   truncation gives 0 but floor should be -1 → subtract 1
    x0_trunc = src_x.to(tl.int32)                              # truncate toward zero
    neg_fix  = (src_x < 0.0).to(tl.int32)                     # 1 if negative, 0 otherwise
    x0_floor = x0_trunc - neg_fix                              # correct floor

    # Clamp to valid source range [0, IN_W-1]
    x0 = tl.maximum(x0_floor,         0)
    x1 = tl.minimum(x0_floor + 1, IN_W - 1)
    x1 = tl.maximum(x1,               0)          # guard against x1=-1 (shouldn't happen)

    # Fractional weight for x1
    wx1 = src_x - x0_floor.to(tl.float32)   # ∈ [0, 1)
    wx0 = 1.0 - wx1

    # Gather sigmoid values at x0 and x1 (IN_W=4 → 4-way select)
    v0 = tl.where(x0 == 0, s0, tl.where(x0 == 1, s1, tl.where(x0 == 2, s2, s3)))
    v1 = tl.where(x1 == 0, s0, tl.where(x1 == 1, s1, tl.where(x1 == 2, s2, s3)))

    interp = v0 * wx0 + v1 * wx1   # [BLOCK_HW] float32

    # Load in_2, multiply, store
    flat_off = c * OUT_HW + hw_offs
    in2 = tl.load(in_2_ptr + flat_off, mask=hw_mask, other=0.0).to(tl.float32)
    result = interp * in2

    if IS_BF16:
        tl.store(out_ptr + flat_off, result.to(tl.bfloat16), mask=hw_mask)
    else:
        tl.store(out_ptr + flat_off, result.to(tl.float16),  mask=hw_mask)


@torch.fx.wrap
def fused_conv_sigmoid_bilinear_mul(in_0, in_1, in_2):
    """
    Fused replacement for:
        conv2d(in_1, in_0) -> sigmoid -> bilinear_upsample(64,128) -> in_2 * ...
    Returns a one-element tuple to match the pattern's return structure.
    """
    N_OC   = 128
    N_IC   = 960
    IN_W   = 4
    OUT_H  = 64
    OUT_W  = 128
    OUT_HW = OUT_H * OUT_W      # 8192
    N_IC_PAD = 1024             # next power-of-2 >= N_IC=960

    # Flatten for the kernel
    in_0_flat = in_0.reshape(N_OC, N_IC)        # [128, 960]
    in_1_flat = in_1.reshape(N_IC, IN_W)         # [960,   4]
    in_2_flat = in_2.reshape(N_OC, OUT_HW)       # [128, 8192]

    out_flat = torch.empty_like(in_2_flat)       # [128, 8192]

    IS_BF16 = (in_2.dtype == torch.bfloat16)

    # Grid depends on the autotuned BLOCK_HW
    grid = lambda meta: (N_OC, triton.cdiv(OUT_HW, meta['BLOCK_HW']))

    _fused_conv_sig_bilinear_mul_kernel[grid](
        in_0_flat, in_1_flat, in_2_flat, out_flat,
        N_OC=N_OC, N_IC=N_IC, IN_W=IN_W,
        OUT_H=OUT_H, OUT_W=OUT_W, OUT_HW=OUT_HW,
        N_IC_PAD=N_IC_PAD,
        IS_BF16=IS_BF16,
    )

    return (out_flat.reshape(1, N_OC, OUT_H, OUT_W),)


def replacement_func():
    return fused_conv_sigmoid_bilinear_mul