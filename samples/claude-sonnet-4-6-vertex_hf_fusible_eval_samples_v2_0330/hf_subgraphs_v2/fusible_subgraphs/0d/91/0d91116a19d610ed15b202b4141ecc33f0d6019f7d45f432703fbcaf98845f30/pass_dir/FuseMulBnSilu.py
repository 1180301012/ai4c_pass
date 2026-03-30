import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: mul -> batch_norm (inference) -> silu
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ---------------------------------------------------------------------------
# Fused Triton kernel
# Grid dim0 = B*C  (each program owns one (batch, channel) slice)
# Grid dim1 = ceil(H*W / BLOCK_HW)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_mul_bn_silu_kernel(
    x_ptr,      # [B, C, H, W]
    sc_ptr,     # [B, C, 1, 1]  (scale = in_4, i.e. sigmoid output)
    mean_ptr,   # [C]  float32
    var_ptr,    # [C]  float32
    w_ptr,      # [C]  float32  (BN weight / gamma)
    b_ptr,      # [C]  float32  (BN bias  / beta)
    out_ptr,    # [B, C, H, W]
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_bc = tl.program_id(0)   # flattened (batch, channel) index
    pid_hw = tl.program_id(1)   # spatial tile index

    b = pid_bc // C
    c = pid_bc % C

    # ---- Load per-channel BN statistics (always fp32) --------------------
    mean_f = tl.load(mean_ptr + c).to(tl.float32)
    var_f  = tl.load(var_ptr  + c).to(tl.float32)
    w_f    = tl.load(w_ptr    + c).to(tl.float32)
    b_f    = tl.load(b_ptr    + c).to(tl.float32)

    # ---- Load scale: element (b, c, 0, 0) in [B, C, 1, 1] ---------------
    sc_f = tl.load(sc_ptr + b * C + c).to(tl.float32)

    # ---- Pre-compute effective affine transform ---------------------------
    # BN(x*sc) = (x*sc - mean)/sqrt(var+eps)*w + bias
    #           = x * (sc * w/sqrt(var+eps)) + (bias - mean*w/sqrt(var+eps))
    eps = 1e-5
    inv_std     = tl.rsqrt(var_f + eps)
    gamma       = w_f * inv_std          # w / sqrt(var+eps)
    gamma_sc    = gamma * sc_f           # absorb the per-(b,c) scale
    beta        = b_f - mean_f * gamma   # bias term

    # ---- Spatial tile offsets --------------------------------------------
    hw_start = pid_hw * BLOCK_HW
    offs     = hw_start + tl.arange(0, BLOCK_HW)
    mask     = offs < HW

    base = (b * C + c) * HW
    x    = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    # ---- Fused: mul(absorbed) + BN + SiLU --------------------------------
    x_f  = x.to(tl.float32)
    y    = x_f * gamma_sc + beta         # BN (with scale absorbed)
    out  = y * tl.sigmoid(y)             # SiLU

    tl.store(out_ptr + base + offs, out.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_mul_bn_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias / beta   [C]
    in_3 : weight / gamma[C]
    in_4 : scale (sigmoid)[B, C, 1, 1]
    in_5 : input x       [B, C, H, W]
    """
    device = in_5.device

    # BN stats / params must be fp32 scalars on the same device
    mean   = in_0.to(device=device, dtype=torch.float32).contiguous()
    var    = in_1.to(device=device, dtype=torch.float32).contiguous()
    weight = in_3.to(device=device, dtype=torch.float32).contiguous()
    bias   = in_2.to(device=device, dtype=torch.float32).contiguous()

    scale  = in_4.contiguous()
    x      = in_5.contiguous()

    B, C, H, W = x.shape
    HW = H * W

    out = torch.empty_like(x)

    grid = lambda meta: (B * C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_mul_bn_silu_kernel[grid](
        x, scale, mean, var, weight, bias, out,
        C, HW,
    )

    return (out,)


def replacement_func():
    return fused_mul_bn_silu