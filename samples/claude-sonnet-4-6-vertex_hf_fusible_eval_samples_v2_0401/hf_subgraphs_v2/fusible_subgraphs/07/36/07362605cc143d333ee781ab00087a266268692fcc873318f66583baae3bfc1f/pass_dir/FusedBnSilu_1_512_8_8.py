"""
Fused kernel: reshape(1,512,8,8) + batch_norm (inference) + silu
Covers both bfloat16 and float16 graphs with 512 channels, 8x8 spatial.

Pattern args:
  in_0 = running_mean  [512]  CPU
  in_1 = running_var   [512]  CPU
  in_2 = bias          [512]  CPU
  in_3 = weight        [512]  CPU
  in_4 = input tensor  [4,128,64]  CUDA  (reshaped to [1,512,8,8] internally)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – must mirror model.py exactly
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Triton kernel – one program per channel, processes HW=64 elements
# ---------------------------------------------------------------------------
@triton.jit
def _fused_bn_silu_512_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps,
    HW: tl.constexpr,   # 64 (= 8 * 8)
):
    """Each program handles exactly one channel (HW contiguous elements)."""
    c    = tl.program_id(0)          # channel index  (0 … C-1)
    base = c * HW

    # ---- scalar loads for per-channel statistics / affine params ----
    mean_val   = tl.load(mean_ptr   + c).to(tl.float32)
    var_val    = tl.load(var_ptr    + c).to(tl.float32)
    weight_val = tl.load(weight_ptr + c).to(tl.float32)
    bias_val   = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale   = weight_val * inv_std
    offset  = bias_val - mean_val * scale

    # ---- vectorised load / compute / store for the HW pixels ----
    offsets = tl.arange(0, HW)
    x = tl.load(x_ptr + base + offsets).to(tl.float32)

    # BN inference
    y = x * scale + offset

    # SiLU:  y * sigmoid(y)
    z = y * tl.sigmoid(y)

    # store – Triton auto-converts float32 → output tensor dtype
    tl.store(out_ptr + base + offsets, z)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_silu_512_8_8_impl(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : running_mean [512]   CPU bfloat16/float16
    in_1 : running_var  [512]   CPU bfloat16/float16
    in_2 : bias         [512]   CPU bfloat16/float16
    in_3 : weight       [512]   CPU bfloat16/float16
    in_4 : input tensor         CUDA bfloat16/float16  (e.g. [4,128,64])
    Returns tensor of shape [1, 512, 8, 8] in the same dtype as in_4.
    """
    N, C, H, W = 1, 512, 8, 8
    HW   = H * W          # 64
    eps  = 1e-5

    device = in_4.device

    # Reshape input to NCHW (contiguous view – no data copy needed)
    x = in_4.reshape(N, C, H, W)
    if not x.is_contiguous():
        x = x.contiguous()

    # Allocate output in the same dtype
    out = torch.empty_like(x)

    # Move BN parameters to GPU (keep native dtype; kernel converts internally)
    mean_gpu   = in_0.to(device=device)
    var_gpu    = in_1.to(device=device)
    weight_gpu = in_3.to(device=device)   # note: in_3 = weight (gamma)
    bias_gpu   = in_2.to(device=device)   # note: in_2 = bias  (beta)

    # One program per channel
    grid = (N * C,)   # = (512,)

    _fused_bn_silu_512_kernel[grid](
        x, mean_gpu, var_gpu, weight_gpu, bias_gpu, out,
        eps,
        HW=HW,
    )

    return out


def replacement_func():
    return fused_bn_silu_512_8_8_impl