"""
Pass: FuseConv2dViewSigmoid_1_2_8_8
Fuses: torch.conv2d(in_2, in_1, in_0, (1,1),(0,0),(1,1),1)
       .view(1,2,8,8).sigmoid()
into a single Triton GEMV+bias+sigmoid kernel.

Conv2d geometry:
  input  in_2: [1, 2, 1, 8]  → flatten → [16]
  weight in_1: [128, 2, 1, 8] → flatten → [128, 16]
  bias   in_0: [128]
  output:       [1, 128, 1, 1] → view(1,2,8,8) → [1, 2, 8, 8]

Because kernel_size == input_spatial_size the conv2d is a pure matrix-vector
multiply (GEMV): y[i] = dot(weight[i,:], input_flat) + bias[i].

One Triton program per output channel (grid 128).  For this tiny geometry a
custom kernel eliminates cuDNN algorithm-selection overhead (typically 10-20 μs
for tiny sizes) and fuses the sigmoid, saving one extra kernel launch.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  — must mirror model.py EXACTLY (positional tuple args for conv2d)
# ---------------------------------------------------------------------------
def pattern(input, weight, bias):
    conv = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    t = conv.view(1, 2, 8, 8)
    out = t.sigmoid()
    return out


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(input, weight, bias):
    return (input, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel
#   Grid: (C_OUT,) = (128,)  — one program per output channel
#   C_IN = 2*1*8 = 16        — elements per dot product
#   Computation: out[i] = sigmoid( dot(input_flat, weight[i,:]) + bias[i] )
# ---------------------------------------------------------------------------
@triton.jit
def conv2d_view_sigmoid_kernel(
    input_ptr,   # [C_IN] = 16 elements (contiguous flat view)
    weight_ptr,  # [C_OUT, C_IN] = [128, 16] elements (contiguous)
    bias_ptr,    # [C_OUT] = 128 elements
    output_ptr,  # [C_OUT] = 128 elements (will be .view(1,2,8,8) outside)
    C_IN: tl.constexpr,   # = 16
):
    out_ch = tl.program_id(0)          # 0 .. C_OUT-1 = 0 .. 127

    in_offs = tl.arange(0, C_IN)      # [0, 1, ..., 15]

    # Load full input once (16 fp16/bf16 elements → float32 for accumulation)
    x = tl.load(input_ptr + in_offs).to(tl.float32)

    # Load weight row for this output channel
    w = tl.load(weight_ptr + out_ch * C_IN + in_offs).to(tl.float32)

    # Dot product
    dot = tl.sum(x * w, axis=0)

    # Add bias
    b = tl.load(bias_ptr + out_ch).to(tl.float32)

    # Sigmoid and store (auto-cast float32 → fp16/bf16 via pointer type)
    tl.store(output_ptr + out_ch, tl.sigmoid(dot + b))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fuse_conv2d_view_sigmoid_1_2_8_8(input, weight, bias):
    """
    input  : [1, 2, 1, 8]  (CUDA, fp16/bf16)
    weight : [128, 2, 1, 8] (CUDA, fp16/bf16)
    bias   : [128]          (CUDA, fp16/bf16)
    returns: [1, 2, 8, 8]   (same dtype/device)
    """
    C_IN  = 16   # 2 * 1 * 8
    C_OUT = 128

    x = input.contiguous().view(-1)        # [16]
    w = weight.contiguous().view(C_OUT, -1) # [128, 16]
    b = bias.contiguous()                   # [128]

    out_flat = torch.empty(C_OUT, dtype=input.dtype, device=input.device)

    conv2d_view_sigmoid_kernel[(C_OUT,)](x, w, b, out_flat, C_IN=C_IN)

    return out_flat.view(1, 2, 8, 8)


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return fuse_conv2d_view_sigmoid_1_2_8_8