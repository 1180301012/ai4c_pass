import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused GELU (exact) + batch-norm inference
# Grid: (C, N, ceil(HW / BLOCK_SIZE))
# Each program handles one (channel, batch) pair for BLOCK_SIZE spatial elements.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
    ],
    key=['HW', 'N', 'C'],
)
@triton.jit
def fused_gelu_bn_kernel(
    in_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    out_gelu_ptr, out_bn_ptr,
    N, C, HW,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c  = tl.program_id(0)   # channel  [0, C)
    pid_n  = tl.program_id(1)   # batch    [0, N)
    pid_hw = tl.program_id(2)   # spatial block

    # ---- Per-channel BN parameters (fp32 for precision) ----
    running_mean = tl.load(running_mean_ptr + pid_c).to(tl.float32)
    running_var  = tl.load(running_var_ptr  + pid_c).to(tl.float32)
    weight_c     = tl.load(weight_ptr       + pid_c).to(tl.float32)
    bias_c       = tl.load(bias_ptr         + pid_c).to(tl.float32)

    # BN inference affine params: y = gelu * scale + offset
    # Use rsqrt (single hardware instruction) instead of 1/sqrt
    inv_std = tl.rsqrt(running_var + eps)
    scale  = weight_c * inv_std
    offset = bias_c - running_mean * scale

    # ---- Spatial offsets ----
    base_offset = (pid_n * C + pid_c) * HW
    offsets_hw  = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask        = offsets_hw < HW
    flat_idx    = base_offset + offsets_hw

    # ---- Load and upcast to fp32 ----
    x = tl.load(in_ptr + flat_idx, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU exact via fast polynomial erf (Abramowitz & Stegun 7.1.26) ----
    # max error < 1.5e-7; uses tl.exp (__expf) which is faster than erff
    INV_SQRT2 = 0.7071067811865476
    z      = x * INV_SQRT2          # x / sqrt(2)
    z_sq   = z * z                  # z^2 == |z|^2, reused for exp
    abs_z  = tl.abs(z)
    t      = 1.0 / (1.0 + 0.3275911 * abs_z)
    poly   = ((((1.061405429 * t
              - 1.453152027) * t
              + 1.421413741) * t
              - 0.284496736) * t
              + 0.254829592) * t
    erf_abs = 1.0 - poly * tl.exp(-z_sq)   # reuse z_sq (no extra abs)
    erf_z   = tl.where(z >= 0.0, erf_abs, -erf_abs)
    gelu_x  = x * 0.5 * (1.0 + erf_z)

    # ---- BN inference: scale * gelu + offset ----
    bn_x = gelu_x * scale + offset

    # ---- Store with dtype cast ----
    if IS_FP16:
        tl.store(out_gelu_ptr + flat_idx, gelu_x.to(tl.float16), mask=mask)
        tl.store(out_bn_ptr   + flat_idx, bn_x.to(tl.float16),   mask=mask)
    elif IS_BF16:
        tl.store(out_gelu_ptr + flat_idx, gelu_x.to(tl.bfloat16), mask=mask)
        tl.store(out_bn_ptr   + flat_idx, bn_x.to(tl.bfloat16),   mask=mask)
    else:
        tl.store(out_gelu_ptr + flat_idx, gelu_x, mask=mask)
        tl.store(out_bn_ptr   + flat_idx, bn_x,   mask=mask)


# ---------------------------------------------------------------------------
# Pattern: match gelu + batch_norm_inference + identity_add
# The pre-summed tensor (result of in_4 += in_5) is treated as an opaque input.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, pre_added):
    """
    in_0      : running_mean  [C]
    in_1      : running_var   [C]
    in_2      : bias          [C]
    in_3      : weight        [C]
    pre_added : already-summed tensor [N, C, H, W]  (= in_4 + in_5)
    """
    tmp_5 = torch.nn.functional.gelu(pre_added, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return tmp_5, tmp_7


# ---------------------------------------------------------------------------
# Argument extractor
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2, in_3, pre_added):
    return (in_0, in_1, in_2, in_3, pre_added)


# ---------------------------------------------------------------------------
# Internal implementation — @torch.fx.wrap makes it an opaque leaf to FX.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_gelu_bn_impl(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias          [C]
    in_3 : weight        [C]
    in_4 : pre-summed activation [N, C, H, W]
    Returns: (gelu_out, bn_out)
    """
    N, C, H, W = in_4.shape
    HW  = H * W
    eps = 1e-5

    IS_FP16 = in_4.dtype == torch.float16
    IS_BF16 = in_4.dtype == torch.bfloat16

    out_gelu = torch.empty_like(in_4)
    out_bn   = torch.empty_like(in_4)

    grid = lambda meta: (C, N, triton.cdiv(HW, meta['BLOCK_SIZE']))

    fused_gelu_bn_kernel[grid](
        in_4,
        in_0, in_1,   # running_mean, running_var
        in_3, in_2,   # weight, bias
        out_gelu, out_bn,
        N, C, HW,
        eps,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )

    return out_gelu, out_bn


# ---------------------------------------------------------------------------
# Outer replacement function — NOT @torch.fx.wrap so FX traces through it.
# FX sees: one opaque call to _fused_gelu_bn_impl + two getitem nodes.
# This yields exactly 2 returning nodes to match the pattern's 2 outputs.
# ---------------------------------------------------------------------------
def fused_gelu_bn(in_0, in_1, in_2, in_3, in_4):
    """
    Replacement function traced by FX.
    Returns (gelu_output, bn_output) as two separate graph nodes.
    """
    result = _fused_gelu_bn_impl(in_0, in_1, in_2, in_3, in_4)
    return result[0], result[1]  # two getitem nodes → 2 returning nodes


# ---------------------------------------------------------------------------
# Replacement function factory (zero-argument, returns callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_gelu_bn