import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: ATen-level ops as produced by torch.compile / Dynamo
#   tmp_2 = aten.relu(in_2)
#   tmp_3 = aten.avg_pool2d(tmp_2, [3,3], [1,1], [1,1], False, False, None)
#   tmp_4 = aten.sub(tmp_3, tmp_2)
#   tmp_5 = aten.unsqueeze(in_0, -1)
#   tmp_6 = aten.unsqueeze(tmp_5, -1)
#   tmp_7 = aten.mul(tmp_6, tmp_4)
#   tmp_8 = aten.add(tmp_2, tmp_7)
# ---------------------------------------------------------------------------
def pattern(in_0, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    return tmp_8


def replacement_args(in_0, in_2):
    return (in_0, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fuses relu + 3x3 avg_pool (count_include_pad=False) +
#                scale * (pool - relu) + relu  into one pass over memory
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def _fused_relu_avgpool_arith_kernel(
    x_ptr,      # [B, C, H, W]  input
    scale_ptr,  # [C]            per-channel scale (in_0)
    out_ptr,    # [B, C, H, W]  output
    B, C, H, W, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # ---- decompose flat index -> (b, c, h, w) ----------------------------
    HW  = H * W
    CHW = C * HW

    w_idx = offsets % W
    h_idx = (offsets // W) % H
    c_idx = (offsets // HW) % C
    b_idx = offsets // CHW

    # base pointer for this (b, c) slice
    base = b_idx * CHW + c_idx * HW

    # ---- load per-channel scale ------------------------------------------
    scale_f32 = tl.load(scale_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # ---- load center pixel and compute relu ------------------------------
    center_raw = tl.load(x_ptr + base + h_idx * W + w_idx, mask=mask, other=0.0)
    center_f32 = center_raw.to(tl.float32)
    relu_center = tl.maximum(center_f32, 0.0)

    # ---- 3×3 avg-pool  (count_include_pad=False) -------------------------
    pool_sum   = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    pool_count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for dh in range(-1, 2):
        for dw in range(-1, 2):
            nh = h_idx + dh
            nw = w_idx + dw
            valid = mask & (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
            nb_raw = tl.load(x_ptr + base + nh * W + nw, mask=valid, other=0.0).to(tl.float32)
            relu_nb = tl.maximum(nb_raw, 0.0)
            pool_sum   = pool_sum   + tl.where(valid, relu_nb, 0.0)
            pool_count = pool_count + tl.where(valid, 1.0,     0.0)

    avg_val = pool_sum / pool_count          # divide by actual neighbor count

    # ---- final formula: relu + scale * (avg - relu) ----------------------
    out_f32 = relu_center + scale_f32 * (avg_val - relu_center)

    # store, casting back to the input dtype
    tl.store(out_ptr + offsets, out_f32.to(center_raw.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_relu_avgpool_arith(in_0, in_2):
    B, C, H, W = in_2.shape
    N = B * C * H * W
    out = torch.empty_like(in_2)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_relu_avgpool_arith_kernel[grid](
        in_2, in_0, out,
        B, C, H, W, N,
    )
    return out


def replacement_func():
    return fused_relu_avgpool_arith