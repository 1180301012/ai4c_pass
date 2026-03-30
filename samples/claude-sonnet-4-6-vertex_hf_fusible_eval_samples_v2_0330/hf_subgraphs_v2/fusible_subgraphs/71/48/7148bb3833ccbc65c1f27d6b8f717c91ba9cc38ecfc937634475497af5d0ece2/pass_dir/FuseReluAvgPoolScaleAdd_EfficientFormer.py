import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: subgraph match on the post-relu portion of the EfficientFormer block.
#
#   Inputs:  relu_out  [N, C, H, W]  – already relu'd feature map
#            in_0      [C]           – per-channel layer_scale
#
#   Computation:
#     avg  = avg_pool2d(relu_out, 3×3, stride=1, pad=1,
#                       count_include_pad=False)
#     out  = relu_out + in_0[:, None, None] * (avg - relu_out)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(relu_out, in_0):
    tmp_3 = torch.nn.functional.avg_pool2d(relu_out, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - relu_out
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = relu_out + tmp_7
    return tmp_8


def replacement_args(relu_out, in_0):
    return (relu_out, in_0)


# ─────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel – flat parallel over all N*C*H*W output elements
#
# Key optimisations vs. naive implementation:
#  • pool_count precomputed analytically (no 9-iteration increment)
#  • masked load with other=0.0 replaces tl.where (saves 9 ops per pixel)
#  • Autotune selects best BLOCK_SIZE / num_warps / num_stages per shape
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_avgpool_scale_kernel(
    relu_out_ptr, in0_ptr, out_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H * W
    mask = offsets < total

    # ── Decompose flat index ──────────────────────────────────────────────────
    tmp_idx = offsets
    w_idx = tmp_idx % W
    tmp_idx = tmp_idx // W
    h_idx = tmp_idx % H
    tmp_idx = tmp_idx // H
    c_idx = tmp_idx % C
    n_idx = tmp_idx // C

    # ── Center pixel ─────────────────────────────────────────────────────────
    # Compute nc_base once; reuse for center_idx AND the pool window loop.
    nc_base    = n_idx * stride_n + c_idx * stride_c
    center_idx = nc_base + h_idx * stride_h + w_idx * stride_w
    x_center = tl.load(relu_out_ptr + center_idx, mask=mask, other=0.0)
    x_f32 = x_center.to(tl.float32)

    # ── Boundary flags (4 comparisons, reused across all 9 loop iterations) ──
    h_at_top    = (h_idx == 0)
    h_at_bottom = (h_idx == H - 1)
    w_at_left   = (w_idx == 0)
    w_at_right  = (w_idx == W - 1)

    # ── pool_count using precomputed flags (avoids 4 max/min ops) ────────────
    h_count = tl.where(h_at_top | h_at_bottom, 2, 3)
    w_count = tl.where(w_at_left  | w_at_right,  2, 3)
    pool_count = (h_count * w_count).to(tl.float32)

    # ── 3×3 average pool (count_include_pad=False) ───────────────────────────
    # Per-neighbour validity is derived from the precomputed flags, not from
    # 4 new comparisons per iteration → saves ~41 ops/element.
    # Center pixel (dh=0, dw=0) is already in x_f32 → reuse, no extra load.
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for dh in range(-1, 2):
        nh      = h_idx + dh
        # Compile-time (dh is a Python int in the unrolled loop)
        if dh == -1:
            nh_valid = ~h_at_top
        elif dh == 1:
            nh_valid = ~h_at_bottom
        else:
            nh_valid = mask   # dh=0: row always in [0, H-1]

        nh_base = nc_base + nh * stride_h

        for dw in range(-1, 2):
            nw = w_idx + dw
            if dw == -1:
                nw_valid = ~w_at_left
            elif dw == 1:
                nw_valid = ~w_at_right
            else:
                nw_valid = mask   # dw=0: col always in [0, W-1]

            if dh == 0 and dw == 0:
                # Centre already in registers; avoids 1 L2 load per element
                pool_sum += x_f32
            else:
                combined = mask & nh_valid & nw_valid
                nb_idx   = nh_base + nw * stride_w
                pool_sum += tl.load(relu_out_ptr + nb_idx, mask=combined, other=0.0).to(tl.float32)

    avg = pool_sum / pool_count

    # ── Scale and output ──────────────────────────────────────────────────────
    scale   = tl.load(in0_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    out_f32 = x_f32 + scale * (avg - x_f32)
    tl.store(out_ptr + center_idx, out_f32.to(x_center.dtype), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_avgpool_scale_add(relu_out, in_0):
    N, C, H, W = relu_out.shape
    out   = torch.empty_like(relu_out)
    total = N * C * H * W

    grid = lambda meta: (
        (total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )

    fused_avgpool_scale_kernel[grid](
        relu_out, in_0, out,
        N, C, H, W,
        relu_out.stride(0), relu_out.stride(1),
        relu_out.stride(2), relu_out.stride(3),
    )
    return out


def replacement_func():
    return fused_avgpool_scale_add