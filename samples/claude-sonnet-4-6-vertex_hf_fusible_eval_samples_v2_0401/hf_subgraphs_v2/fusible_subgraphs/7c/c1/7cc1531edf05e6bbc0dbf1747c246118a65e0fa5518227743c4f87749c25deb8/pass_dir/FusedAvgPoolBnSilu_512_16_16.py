"""
Fused pass: reshape + avg_pool2d(2,2,stride=2) + batch_norm(inference) + silu
Target shapes:
  in_4  : [4, 128, 256]  -> viewed as [1, 512, 16, 16]
  in_0  : [512]  running_mean
  in_1  : [512]  running_var
  in_2  : [512]  bias
  in_3  : [512]  weight
  output: [1, 512, 8, 8]

Strategy:
  Grid (512,) — one program per channel.
  Each program handles all 64 spatial output elements (8x8) in one block.
  Compute in float32 for numerical accuracy, store in input dtype.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Triton fused kernel
# ---------------------------------------------------------------------------
@triton.jit
def fused_avgpool_bn_silu_kernel(
    input_ptr,   # contiguous [1, 512, 16, 16] (alias of [4,128,256])
    mean_ptr,    # running_mean  [512]  (in_0)
    var_ptr,     # running_var   [512]  (in_1)
    bias_ptr,    # bias          [512]  (in_2)
    weight_ptr,  # weight        [512]  (in_3)
    output_ptr,  # output        [1, 512, 8, 8]
    BLOCK_SPATIAL: tl.constexpr,   # = 64  (8*8)
):
    """
    Each program handles one channel (program_id(0) = channel index c).
    Covers all BLOCK_SPATIAL=64 spatial output positions in one block.

    Input layout  [1, 512, 16, 16]: offset(0,c,h,w) = c*256 + h*16 + w
    Output layout [1, 512,  8,  8]: offset(0,c,h,w) = c*64  + h*8  + w
    """
    c = tl.program_id(0)

    # All 64 spatial output indices
    spatial = tl.arange(0, BLOCK_SPATIAL)   # [0 .. 63]
    h_out = spatial // 8                    # row in [0,7]
    w_out = spatial % 8                     # col in [0,7]

    # Base pointer into the 2x2 input window for each output position
    # offset = c*256 + (h_out*2)*16 + (w_out*2)
    #        = c*256 + h_out*32 + w_out*2
    in_base = c * 256 + h_out * 32 + w_out * 2

    # Load 4 elements of the 2x2 pooling window and upcast to fp32
    x00 = tl.load(input_ptr + in_base).to(tl.float32)       # (h*2,   w*2  )
    x01 = tl.load(input_ptr + in_base + 1).to(tl.float32)   # (h*2,   w*2+1)
    x10 = tl.load(input_ptr + in_base + 16).to(tl.float32)  # (h*2+1, w*2  )
    x11 = tl.load(input_ptr + in_base + 17).to(tl.float32)  # (h*2+1, w*2+1)

    # Average pooling (count_include_pad=True; interior window, no edge effect)
    avg = (x00 + x01 + x10 + x11) * 0.25

    # Load BN parameters for channel c (upcast to fp32 for accuracy)
    mean    = tl.load(mean_ptr   + c).to(tl.float32)
    var     = tl.load(var_ptr    + c).to(tl.float32)
    weight  = tl.load(weight_ptr + c).to(tl.float32)
    bias_v  = tl.load(bias_ptr   + c).to(tl.float32)

    # Batch norm inference: y = (x - mean) / sqrt(var + eps) * weight + bias
    #  Precompute scale/shift to save ops
    inv_std = tl.rsqrt(var + 1e-05)
    scale   = weight * inv_std
    shift   = bias_v - mean * scale
    bn_out  = avg * scale + shift

    # SiLU: x * sigmoid(x)
    silu_out = bn_out * tl.sigmoid(bn_out)

    # Store output at [0, c, h_out, w_out] = c*64 + spatial
    # Triton auto-casts float32 -> output pointer dtype (bf16 / fp16)
    tl.store(output_ptr + c * 64 + spatial, silu_out)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    """
    Drop-in replacement for:
        reshape + avg_pool2d + batch_norm(inference) + silu

    Args
    ----
    in_0 : running_mean [512]  (may be on CPU)
    in_1 : running_var  [512]  (may be on CPU)
    in_2 : bias         [512]  (may be on CPU)
    in_3 : weight       [512]  (may be on CPU)
    in_4 : activations  [4, 128, 256]  on CUDA

    Returns
    -------
    Tensor [1, 512, 8, 8] on same device/dtype as in_4
    """
    device = in_4.device
    dtype  = in_4.dtype

    # Ensure BN statistics and learnable params are on the compute device
    mean   = in_0.to(device=device)
    var    = in_1.to(device=device)
    bias   = in_2.to(device=device)
    weight = in_3.to(device=device)

    # Guarantee contiguous layout (required for our stride assumptions)
    x = in_4.contiguous()

    # Allocate output
    output = torch.empty(1, 512, 8, 8, device=device, dtype=dtype)

    C             = 512
    BLOCK_SPATIAL = 64   # 8 * 8

    fused_avgpool_bn_silu_kernel[(C,)](
        x,
        mean,
        var,
        bias,
        weight,
        output,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    return output


# ---------------------------------------------------------------------------
# Replacement function (returns the callable, does NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_avgpool_bn_silu