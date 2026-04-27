import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: batch_norm (inference) + prelu → fused into one Triton kernel.
#
# Kernel design:
#   - 2-D grid: dim-0 = (n,c) pair, dim-1 = spatial tile
#   - BLOCK_SIZE=512, num_warps=2  →  64 threads/block
#     → 512/64 = 8 fp16 elements per thread → 128-bit vectorised loads ✓
#     → 32 max-warps / 2 warps-per-block = 16 concurrent blocks per SM
#     → 56 SMs × 16 blocks = 896 concurrent blocks (vs 224 with num_warps=8)
#   - No autotune: one JIT compilation per dtype, no calibration overhead.
#   - No loop: each program handles exactly BLOCK_SIZE elements (with masking
#     for the last tile when HW is not a multiple of BLOCK_SIZE).
# -----------------------------------------------------------------------

def pattern(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    bn  = torch.nn.functional.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001)
    out = torch.prelu(bn, prelu_weight)
    return out


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


@triton.jit
def fused_bn_prelu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    prelu_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    nc_idx    = tl.program_id(0)   # which (n, c) pair
    block_idx = tl.program_id(1)   # which spatial tile
    c         = nc_idx % C

    # Per-channel BN / PReLU parameters → fp32 for accuracy
    mean  = tl.load(mean_ptr   + c).to(tl.float32)
    var   = tl.load(var_ptr    + c).to(tl.float32)
    gamma = tl.load(weight_ptr + c).to(tl.float32)
    beta  = tl.load(bias_ptr   + c).to(tl.float32)
    pw    = tl.load(prelu_ptr  + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 1e-3)
    scale   = gamma * inv_std
    shift   = beta - mean * scale

    hw_start = block_idx * BLOCK_SIZE
    offsets  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW

    base = nc_idx * HW
    x   = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    y   = x * scale + shift
    out = tl.where(y >= 0.0, y, pw * y)
    tl.store(out_ptr + base + offsets, out, mask=mask)


@torch.fx.wrap
def fused_bn_prelu(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    N, C, H, W = x.shape
    HW     = H * W
    num_nc = N * C
    out    = torch.empty_like(x)

    # BLOCK_SIZE=512, num_warps=2:
    #   • 64 threads × 8 fp16 per thread → 1 × 128-bit load (perfect vectorisation)
    #   • 32 max-warps / 2 warps-per-block = 16 concurrent blocks per SM
    #     → 56 SMs × 16 = 896 concurrent blocks
    #   • With HW=1024: 2 exact tiles (no masking) → 32768 programs, 36.6 waves
    #   • With HW=784:  2 tiles, last 53% valid     →  8192 programs,  9.1 waves
    #   • With HW=2304: 5 tiles, last 50% valid     → optimal for N=128 cases
    #   2-warp latency hiding: while warp-0 waits for memory, warp-1 computes.
    num_blocks = (HW + 511) // 512

    fused_bn_prelu_kernel[(num_nc, num_blocks)](
        x, running_mean, running_var, bn_weight, bn_bias, prelu_weight,
        out,
        C, HW,
        BLOCK_SIZE=512,
        num_warps=2,
    )
    return out


def replacement_func():
    return fused_bn_prelu