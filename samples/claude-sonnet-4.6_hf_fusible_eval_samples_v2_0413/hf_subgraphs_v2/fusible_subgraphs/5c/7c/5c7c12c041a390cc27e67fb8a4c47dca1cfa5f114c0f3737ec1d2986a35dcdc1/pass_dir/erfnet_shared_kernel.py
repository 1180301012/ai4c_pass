"""
Shared fused Triton kernel for ERFNet's pool+cat+BN+ReLU pattern.

Fuses: max_pool2d(2x2, s=2) + [no-op interpolate] + cat + BN(inference) + ReLU

tensor_a : [N, C_A, H, W]   -- first tensor in cat (already at right spatial size)
tensor_b : [N, C_B, 2H, 2W] -- second tensor in cat (needs 2x2 max-pool, output already = H×W)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N', 'C_A', 'C_B', 'H', 'W'],
)
@triton.jit
def _fused_pool_cat_bn_relu_kernel(
    tensor_a_ptr,   # [N, C_A, H, W]
    tensor_b_ptr,   # [N, C_B, H*2, W*2]
    bn_mean_ptr,    # [C_total]
    bn_var_ptr,     # [C_total]
    bn_weight_ptr,  # [C_total]
    bn_bias_ptr,    # [C_total]
    out_ptr,        # [N, C_total, H, W]
    N, C_A, C_B, H, W, C_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    HW = H * W
    total = N * C_total * HW
    mask = offsets < total

    # Decode flat index -> (n, c, h, w)
    n      = offsets // (C_total * HW)
    rem    = offsets % (C_total * HW)
    c      = rem // HW
    hw_idx = rem % HW
    h      = hw_idx // W
    w      = hw_idx % W

    is_in_a = c < C_A

    # ---- Load from tensor_a (channels 0 .. C_A-1) ----
    a_idx = n * (C_A * HW) + c * HW + hw_idx
    val_a = tl.load(tensor_a_ptr + a_idx,
                    mask=mask & is_in_a, other=0.0).to(tl.float32)

    # ---- Load 2x2 max-pool from tensor_b (channels C_A .. C_total-1) ----
    # Use safe index (clamp to 0) so all memory accesses are within bounds;
    # the is_in_a mask controls what actually gets used.
    c_b   = tl.where(is_in_a, 0, c - C_A)
    W2    = W * 2
    H2    = H * 2
    h2    = h * 2
    w2    = w * 2

    b_base = n * (C_B * H2 * W2) + c_b * (H2 * W2)
    b_mask = mask & ~is_in_a

    v00 = tl.load(tensor_b_ptr + b_base + h2 * W2 + w2,
                  mask=b_mask, other=0.0).to(tl.float32)
    v01 = tl.load(tensor_b_ptr + b_base + h2 * W2 + w2 + 1,
                  mask=b_mask, other=0.0).to(tl.float32)
    v10 = tl.load(tensor_b_ptr + b_base + (h2 + 1) * W2 + w2,
                  mask=b_mask, other=0.0).to(tl.float32)
    v11 = tl.load(tensor_b_ptr + b_base + (h2 + 1) * W2 + w2 + 1,
                  mask=b_mask, other=0.0).to(tl.float32)
    val_b = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    val = tl.where(is_in_a, val_a, val_b)

    # ---- Batch-norm (inference mode, eps=0.001) ----
    mean   = tl.load(bn_mean_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(bn_var_ptr    + c, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(bn_weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bn_bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    normalized = weight * (val - mean) * tl.rsqrt(var + 0.001) + bias

    # ---- ReLU ----
    result = tl.maximum(normalized, 0.0)

    # ---- Store (auto-cast to output dtype) ----
    out_idx = n * (C_total * HW) + c * HW + hw_idx
    tl.store(out_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def fused_pool_cat_bn_relu(tensor_a, tensor_b, bn_mean, bn_var, bn_weight, bn_bias):
    """
    Fused: max_pool2d(tensor_b, 2,2) -> cat([tensor_a, pooled_b], 1) -> BN -> ReLU

    tensor_a : [N, C_A, H,   W]
    tensor_b : [N, C_B, H*2, W*2]
    Returns  : [N, C_A+C_B, H, W]
    """
    N, C_A, H, W = tensor_a.shape
    C_B    = tensor_b.shape[1]
    C_total = C_A + C_B

    out = torch.empty((N, C_total, H, W),
                      dtype=tensor_a.dtype, device=tensor_a.device)

    total = N * C_total * H * W
    grid  = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_pool_cat_bn_relu_kernel[grid](
        tensor_a, tensor_b,
        bn_mean, bn_var, bn_weight, bn_bias,
        out,
        N, C_A, C_B, H, W, C_total,
    )

    return out