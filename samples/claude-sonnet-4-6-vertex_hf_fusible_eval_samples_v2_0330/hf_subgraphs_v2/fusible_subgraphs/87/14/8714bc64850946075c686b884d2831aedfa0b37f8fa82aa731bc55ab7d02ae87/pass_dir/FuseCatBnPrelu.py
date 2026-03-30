import torch
import triton
import triton.language as tl


def pattern(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    """
    Match: cat([x1, x2], dim=1) → BatchNorm (inference) → PReLU

    Fusing all three eliminates the intermediate [N, 2C, H, W] tensor that
    cat would write and BN would immediately re-read.
    """
    cat_out  = torch.cat([x1, x2], 1)
    bn_out   = torch.nn.functional.batch_norm(cat_out, running_mean, running_var,
                                               bn_weight, bn_bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


@triton.autotune(
    configs=[
        # Small BLOCK_HW → high grid parallelism; good for small NC (bfloat16/N=1)
        triton.Config({'BLOCK_HW': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=2),
        # Larger BLOCK_HW; good for medium NC (N=32) and large NC (N=128)
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=2),
        # High-warp configs; best for large NC (N=128, NC=16384)
        triton.Config({'BLOCK_HW': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=2),
    ],
    key=['HW', 'NC'],
)
@triton.jit
def _cat_bn_prelu_kernel(
    x1_ptr, x2_ptr, out_ptr,
    running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    prelu_weight_ptr,
    C_HALF, C_FULL, HW,
    NC,              # N * C_FULL — autotune key only, not used in body
    BLOCK_HW: tl.constexpr,
):
    """
    3-D grid: (N, C_FULL, ceil(HW / BLOCK_HW))
      pid_n  → sample index
      pid_c  → full channel index [0, C_FULL)
      pid_hw → spatial block

    Reads directly from x1 (channels 0..C_HALF-1) or x2 (channels C_HALF..C_FULL-1)
    without ever materialising the concatenated tensor.
    """
    pid_n  = tl.program_id(0)   # sample index
    pid_c  = tl.program_id(1)   # full channel (0 .. C_FULL-1)
    pid_hw = tl.program_id(2)   # spatial block

    # ---- BN + PReLU parameters for this channel (scalar loads) ----
    mean    = tl.load(running_mean_ptr + pid_c).to(tl.float32)
    var     = tl.load(running_var_ptr  + pid_c).to(tl.float32)
    bn_w    = tl.load(bn_weight_ptr    + pid_c).to(tl.float32)
    bn_b    = tl.load(bn_bias_ptr      + pid_c).to(tl.float32)
    prelu_w = tl.load(prelu_weight_ptr + pid_c).to(tl.float32)

    eps     = 0.001
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = bn_w * inv_std
    offset  = bn_b - bn_w * mean * inv_std

    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW

    # ---- Select source: x1 (first half) or x2 (second half) ----
    # Both x1 and x2 have shape [N, C_HALF, H, W].
    # The source channel index within either tensor is pid_c % C_HALF.
    c_src    = pid_c % C_HALF                          # channel within x1 or x2
    src_base = (pid_n * C_HALF + c_src) * HW           # same formula for both tensors

    in_x1    = (pid_c < C_HALF)                        # scalar boolean
    in_x2    = (pid_c >= C_HALF)

    # Masked loads: each pointer is only dereferenced for the correct half.
    # `other=0.0` ensures the wrong half contributes exactly 0.
    x1_vals = tl.load(x1_ptr + src_base + hw_off, mask=mask & in_x1, other=0.0)
    x2_vals = tl.load(x2_ptr + src_base + hw_off, mask=mask & in_x2, other=0.0)
    x_raw   = x1_vals + x2_vals   # exactly one of them is non-zero

    # ---- Fused BN (inference) + PReLU ----
    x_fp32 = x_raw.to(tl.float32)
    y      = scale * x_fp32 + offset
    z      = tl.where(y >= 0.0, y, prelu_w * y)

    # ---- Store output [N, C_FULL, H, W] ----
    out_base = (pid_n * C_FULL + pid_c) * HW
    tl.store(out_ptr + out_base + hw_off, z.to(x_raw.dtype), mask=mask)


@torch.fx.wrap
def cat_bn_prelu_fused(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    """
    Fused cat([x1, x2], dim=1) + BatchNorm (inference) + PReLU.

    Returns a tensor of shape [N, 2*C, H, W] and same dtype as x1/x2.
    """
    N, C_HALF, H, W = x1.shape
    C_FULL = 2 * C_HALF
    HW     = H * W
    NC     = N * C_FULL

    out  = torch.empty((N, C_FULL, H, W), dtype=x1.dtype, device=x1.device)
    grid = lambda meta: (N, C_FULL, triton.cdiv(HW, meta['BLOCK_HW']))

    _cat_bn_prelu_kernel[grid](
        x1, x2, out,
        running_mean, running_var,
        bn_weight, bn_bias, prelu_weight,
        C_HALF, C_FULL, HW, NC,
    )

    return out


def replacement_func():
    return cat_bn_prelu_fused