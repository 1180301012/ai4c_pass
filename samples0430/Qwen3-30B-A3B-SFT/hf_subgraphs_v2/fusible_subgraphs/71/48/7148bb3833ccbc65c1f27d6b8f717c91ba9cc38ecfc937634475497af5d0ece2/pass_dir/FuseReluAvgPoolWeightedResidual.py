import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'TILE_HW': 64}, num_warps=4),
        triton.Config({'TILE_HW': 128}, num_warps=4),
        triton.Config({'TILE_HW': 256}, num_warps=8),
        triton.Config({'TILE_HW': 512}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def _fused_relu_avgpool_residual_kernel(
    in2_ptr,       # [B*C*H*W] contiguous input
    scale1_ptr,    # [C] channel scale (in_0)
    scale2_ptr,    # [C] channel scale (in_1)
    out_ptr,       # [B*C*H*W] output
    B, C, H, W,
    TILE_HW: tl.constexpr,
):
    """
    Fused: relu + avg_pool2d(kernel=3, stride=1, pad=1) + weighted residual add
    Output: relu_x + scale1 * (avgpool(relu_x) - relu_x)

    Grid: (B*C, ceil(H*W / TILE_HW))
    Each program processes TILE_HW pixels from one (b, c) channel-slice.
    Pixel (h, w) is identified by flat index hw = h*W + w.
    """
    pid_bc = tl.program_id(0)   # which (b, c) slice
    pid_hw = tl.program_id(1)   # which tile of H*W

    c = pid_bc % C

    # Load per-channel scale factors
    scale1 = tl.load(scale1_ptr + c)
    scale2 = tl.load(scale2_ptr + c)

    # Spatial tile
    hw_start = pid_hw * TILE_HW
    hw_offsets = hw_start + tl.arange(0, TILE_HW)
    hw_mask = hw_offsets < H * W

    # Convert flat hw -> (h, w)
    h = hw_offsets // W
    w = hw_offsets % W

    # Base pointer for this (b, c) slice
    base = pid_bc * H * W

    # ── Load 9 neighbors (with boundary masking) ────────────────────────────
    # v00  v01  v02
    # v10  v11  v12
    # v20  v21  v22
    v00 = tl.load(in2_ptr + base + (h - 1) * W + (w - 1),
                  mask=hw_mask & (h > 0) & (w > 0), other=0.0)
    v01 = tl.load(in2_ptr + base + (h - 1) * W + w,
                  mask=hw_mask & (h > 0), other=0.0)
    v02 = tl.load(in2_ptr + base + (h - 1) * W + (w + 1),
                  mask=hw_mask & (h > 0) & (w < W - 1), other=0.0)
    v10 = tl.load(in2_ptr + base + h * W + (w - 1),
                  mask=hw_mask & (w > 0), other=0.0)
    v11 = tl.load(in2_ptr + base + h * W + w, mask=hw_mask, other=0.0)
    v12 = tl.load(in2_ptr + base + h * W + (w + 1),
                  mask=hw_mask & (w < W - 1), other=0.0)
    v20 = tl.load(in2_ptr + base + (h + 1) * W + (w - 1),
                  mask=hw_mask & (h < H - 1) & (w > 0), other=0.0)
    v21 = tl.load(in2_ptr + base + (h + 1) * W + w,
                  mask=hw_mask & (h < H - 1), other=0.0)
    v22 = tl.load(in2_ptr + base + (h + 1) * W + (w + 1),
                  mask=hw_mask & (h < H - 1) & (w < W - 1), other=0.0)

    # ── Per-element computation ──────────────────────────────────────────────
    # ReLU
    relu_v11 = tl.maximum(v11, 0.0)

    # Valid neighbor count for denominator (count_include_pad = False)
    count = (
        tl.sum((h > 0) & (w > 0),       axis=0) +
        tl.sum((h > 0),                  axis=0) +
        tl.sum((h > 0) & (w < W - 1),   axis=0) +
        tl.sum((w > 0),                  axis=0) +
        tl.sum(hw_mask) +
        tl.sum((w < W - 1),              axis=0) +
        tl.sum((h < H - 1) & (w > 0),   axis=0) +
        tl.sum((h < H - 1),              axis=0) +
        tl.sum((h < H - 1) & (w < W - 1), axis=0)
    )

    # Average pool (divide by actual valid count, matching count_include_pad=False)
    pool = (v00 + v01 + v02 + v10 + relu_v11 + v12 + v20 + v21 + v22) / count

    # Weighted residual add: relu_x + scale1 * (avgpool - relu_x)
    result = relu_v11 + scale1 * (pool - relu_v11)

    # ── Store ────────────────────────────────────────────────────────────────
    tl.store(out_ptr + base + hw_offsets, result, mask=hw_mask)


# ── Pattern ───────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    relu_out = torch.nn.functional.relu(in_2, inplace=False)
    pool_out = torch.nn.functional.avg_pool2d(relu_out, 3, 1, 1, False, False, None)
    diff = pool_out - relu_out
    scaled = in_0.unsqueeze(-1).unsqueeze(-1) * diff
    out = relu_out + scaled
    out2 = in_1.unsqueeze(-1).unsqueeze(-1)
    return (out, out2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ── Replacement kernel wrapper ────────────────────────────────────────────────
@torch.fx.wrap
def _fused_relu_avgpool_residual(in_0, in_1, in_2):
    B = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]

    out = torch.empty_like(in_2)

    # in_0 and in_1 are [C] channel-wise scale factors; unsqueeze(-1).unsqueeze(-1)
    # produces [C, 1, 1] views without copying data.
    # We pass them directly to the kernel (kernel uses c = pid_bc % C as index).

    grid = lambda meta: (B * C, triton.cdiv(H * W, meta['TILE_HW']))

    _fused_relu_avgpool_residual_kernel[grid](
        in_2, in_0, in_1, out,
        B, C, H, W,
    )

    out2 = in_1.reshape(C, 1, 1)
    return (out, out2)


def replacement_func():
    return _fused_relu_avgpool_residual