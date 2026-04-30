import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full subgraph in model.py
# in_4 += in_5  →  gelu  →  batch_norm (inference)  →  0 + .  →  return
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    in_4 += in_5
    in_6 = in_4
    tmp_5 = torch.nn.functional.gelu(in_6, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ---------------------------------------------------------------------------
# Fused Triton kernel: Add + GELU + BatchNorm (inference) + identity(0+x)
#
# BN inference formula:
#   y = (x - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c]
#     = x * scale[c] + shift[c]
# where scale[c] = weight[c] / sqrt(var[c] + eps)
#       shift[c] = bias[c] - mean[c] * scale[c]
#
# GELU (exact, approximate='none'):
#   gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _fused_add_gelu_bn_kernel(
    x_ptr,          # [N, C, H, W] first addend
    y_ptr,          # [N, C, H, W] second addend
    mean_ptr,       # [C]          running_mean
    var_ptr,        # [C]          running_var
    weight_ptr,     # [C]          bn weight (gamma)
    bias_ptr,       # [C]          bn bias  (beta)
    gelu_out_ptr,   # [N, C, H, W] gelu output
    bn_out_ptr,     # [N, C, H, W] bn output
    C,              # number of channels
    HW,             # H * W (spatial size)
    eps,            # batch-norm epsilon
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C * HW
    mask = offsets < total

    # Channel index for each lane
    c_idx = (offsets // HW) % C

    # ----- Load inputs -------------------------------------------------------
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # ----- Add ---------------------------------------------------------------
    z = x + y

    # ----- GELU (exact) ------------------------------------------------------
    gelu = z * 0.5 * (1.0 + tl.math.erf(z * 0.7071067811865476))

    # ----- Batch-Norm (inference): load per-channel stats -------------------
    mean_val   = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    var_val    = tl.load(var_ptr    + c_idx, mask=mask, other=1.0).to(tl.float32)
    w_val      = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b_val      = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    # Pre-compute fused scale/shift
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale   = w_val * inv_std
    shift   = b_val - mean_val * scale

    # ----- Apply BN ----------------------------------------------------------
    bn = gelu * scale + shift

    # ----- Store (0 + bn ≡ bn) -----------------------------------------------
    dtype = z.dtype  # preserve original dtype
    tl.store(gelu_out_ptr + offsets, gelu.to(dtype), mask=mask)
    tl.store(bn_out_ptr   + offsets, bn.to(dtype),   mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX doesn't trace inside it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_gelu_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias (beta)   [C]
    in_3 : weight (gamma)[C]
    in_4 : activation A  [N, C, H, W]
    in_5 : activation B  [N, C, H, W]

    Returns (gelu_out, bn_out) matching the model's (tmp_5, tmp_7).
    """
    N, C, H, W = in_4.shape
    HW    = H * W
    total = N * C * HW
    eps   = 1e-5

    gelu_out = torch.empty_like(in_4)
    bn_out   = torch.empty_like(in_4)

    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_add_gelu_bn_kernel[grid](
        in_4, in_5,
        in_0, in_1, in_3, in_2,   # mean, var, weight(gamma), bias(beta)
        gelu_out, bn_out,
        C, HW, eps,
    )

    return (gelu_out, bn_out)


# ---------------------------------------------------------------------------
# replacement_func: return the wrapper (zero-argument, returns callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_add_gelu_bn