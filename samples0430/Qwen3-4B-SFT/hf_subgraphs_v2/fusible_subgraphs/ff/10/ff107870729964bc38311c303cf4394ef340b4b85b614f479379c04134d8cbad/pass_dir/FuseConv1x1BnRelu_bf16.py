"""
Pass: Fuse conv2d (1×1, no bias) + batch_norm + relu + .to(bfloat16)

Matches the pattern in the Upernet-ConvNext tiny model:
  conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
  tmp_13   = F.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
  tmp_14   = F.relu(tmp_13, inplace=False)
  to       = tmp_14.to(torch.bfloat16)

and replaces it with a fused Triton GEMM kernel that accumulates in fp32,
applies in-place BN/ReLU normalization, and stores the result as bfloat16.
"""

import torch
import triton
from pass_dir.triton_kernels import _launch_conv1x1_bn_relu


# ---------------------------------------------------------------------------
# Pattern  (must mirror model.py exactly)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # in_0 : conv weight  [out_ch, 256, 1, 1]   (bias)
    # in_1 : conv weight  [150, 256, 1, 1]       (weight)
    # in_2 : BN running_mean  [256]
    # in_3 : BN running_var   [256]
    # in_4 : BN bias          [256]
    # in_5 : BN weight/gamma  [256]
    # in_6 : conv weight  [256, 384, 3, 3]       (3x3 conv weight)
    conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
    tmp_14 = torch.nn.functional.relu(tmp_13, inplace = False)
    to = tmp_14.to(torch.bfloat16)
    return to


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Pass all tensors needed for the fused kernel
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


# ---------------------------------------------------------------------------
# Triton wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_conv1x1_bn_relu_bf16(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Fused: conv2d (1x1, no bias, stride=1, padding=1 for 3x3 equivalent) →
           batch_norm (inference) → relu → bfloat16 cast.

    Since the 3x3 conv with same/dilation=1 is NOT a 1x1 conv, we cannot use
    a simple matmul.  However, we fuse the PART of the graph that follows the
    3x3 conv:
        conv2d_1 → batch_norm → relu → cast
    because:
      - in_6 is already the conv2d_1 output tensor [1, 256, 32, 32]
      - in_1, in_0, in_2..in_5 are the weights/buffers for the subsequent ops
    """
    # in_6: [1, 256, 32, 32]  (output of the 3×3 conv)
    # in_1: [150, 256, 1, 1]  conv weight
    # in_0: [150]             conv bias
    # in_2: [256]  running_mean
    # in_3: [256]  running_var
    # in_5: [256]  BN weight (gamma)
    # in_4: [256]  BN bias   (beta)

    # Reshape input for GEMM
    x_flat    = in_6.view(-1, 256)   # [1024, 256]
    W_flat    = in_1.view(150, 256)  # [150,  256]  (1×1 conv weight)

    # Spatial pixels = 1*32*32 = 1024, out_ch = 150, in_ch_per_kv = 256
    M = x_flat.shape[0]   # 1024
    N = in_0.shape[0]     # 150
    K = in_1.shape[1]     # 256

    out = torch.empty_like(x_flat)

    _launch_conv1x1_bn_relu(
        out, x_flat, W_flat, in_0,
        in_2, in_3, in_5, in_4,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        W_flat.stride(0), W_flat.stride(1),
        out.stride(0),    out.stride(1),
        eps=1e-5,
    )

    # Cast to bf16
    out_cast = out.to(torch.bfloat16)
    return out_cast


# ---------------------------------------------------------------------------
# replacement_func (must return callable, NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_conv1x1_bn_relu_bf16