import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: fuse  scale*x + bias  +  cat([pool_out, scale*x+bias], dim=1)
# By using relu_out as a placeholder, we avoid the ForceArgsTracer kwarg issue.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, tmp_2, tmp_5):
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return tmp_6


def replacement_args(in_0, in_1, tmp_2, tmp_5):
    return (in_0, in_1, tmp_2, tmp_5)


# ──────────────────────────────────────────────────────────────────────────────
# Fused kernel: 2D grid  (blocks_per_batch × B)
#   • b = program_id(1)  → selects which batch element
#   • pid = program_id(0) → selects which BLOCK_SIZE chunk within that batch
#
# Memory layout (contiguous tensors):
#   tensor[b, c, h, w]  at flat offset  b*(C*H*W) + c*(H*W) + h*W + w
#   n_per_batch = C*H*W   (elements per batch)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_per_batch'],
)
@triton.jit
def scale_bias_cat_kernel(
    relu_ptr,
    pool_ptr,
    bias_ptr,
    scale_ptr,
    out_ptr,
    n_per_batch,      # C * H * W
    out_stride_b,     # 2 * C * H * W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)   # block within batch
    b   = tl.program_id(1)   # batch index

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_per_batch

    # ── Scalar bias / scale ───────────────────────────────────────────────────
    bias  = tl.load(bias_ptr).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    # ── Pool output → out[:, :C, :, :] ───────────────────────────────────────
    pool_base = b * n_per_batch
    pool_val  = tl.load(pool_ptr + pool_base + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + b * out_stride_b + offsets,
             pool_val.to(tl.load(pool_ptr).dtype), mask=mask)

    # ── relu_out * scale + bias → out[:, C:, :, :] ────────────────────────────
    relu_base = b * n_per_batch
    relu_val  = tl.load(relu_ptr + relu_base + offsets, mask=mask, other=0.0).to(tl.float32)
    out_val   = relu_val * scale + bias
    tl.store(out_ptr + b * out_stride_b + n_per_batch + offsets,
             out_val.to(tl.load(relu_ptr).dtype), mask=mask)


@torch.fx.wrap
def fused_scale_bias_cat(in_0, in_1, tmp_2, tmp_5):
    """
    in_0  = bias  [1]
    in_1  = scale [1]
    tmp_2 = relu output   [B, C, H, W]
    tmp_5 = pool output   [B, C, H, W]
    Returns: [B, 2C, H, W] = cat([tmp_5, relu*scale+bias], dim=1)
    """
    B, C, H, W = tmp_2.shape

    out = torch.empty((B, 2 * C, H, W), dtype=tmp_2.dtype, device=tmp_2.device)

    n_per_batch  = C * H * W   # elements per batch (scalar)
    out_stride_b = 2 * C * H * W  # batch stride for output

    # 2D grid: (blocks_per_batch, B)
    scale_bias_cat_kernel[
        lambda meta: (triton.cdiv(n_per_batch, meta['BLOCK_SIZE']), B)
    ](
        tmp_2, tmp_5, in_0, in_1, out,
        n_per_batch, out_stride_b,
    )

    return out


def replacement_func():
    return fused_scale_bias_cat