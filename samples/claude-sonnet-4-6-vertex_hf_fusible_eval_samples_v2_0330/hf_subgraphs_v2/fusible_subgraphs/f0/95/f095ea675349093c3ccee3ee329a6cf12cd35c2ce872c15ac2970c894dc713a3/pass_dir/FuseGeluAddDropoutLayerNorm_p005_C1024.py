"""
Fused pass for: GELU + transpose + add + dropout(inference) + LayerNorm
Targets the bfloat16 graph with dropout rate 0.05.

Pattern:
  tmp_5 = gelu(tmp_4)            # [B, C, T]
  tmp_6 = tmp_5.transpose(1, 2)  # [B, T, C]
  tmp_7 = in_3 + tmp_6           # [B, T, C]
  tmp_8 = dropout(tmp_7, 0.05, training=False)  # identity
  tmp_10 = layer_norm(tmp_8, (1024,), weight, bias, 1e-5)
  return (tmp_8, tmp_10)

The fusion avoids multiple round-trips through HBM by doing everything
in a single kernel: one program per (batch, seq) row processes C=1024 elements.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["C"],
)
@triton.jit
def _fused_gelu_add_layernorm_kernel(
    # Inputs
    tmp4_ptr,    # [B, C, T_slice]  – gelu input, may be non-contiguous
    in3_ptr,     # [B, T, C]        – residual, contiguous
    weight_ptr,  # [C]              – LayerNorm weight
    bias_ptr,    # [C]              – LayerNorm bias
    # Outputs
    out1_ptr,    # [B, T, C]        – tmp_8 (post-add, pre-layernorm)
    out2_ptr,    # [B, T, C]        – tmp_10 (post-layernorm)
    # Dims
    B, T, C,
    # Strides for tmp4 [B, C, T_slice]
    tmp4_s0, tmp4_s1, tmp4_s2,
    # Strides for in3 [B, T, C]
    in3_s0, in3_s1, in3_s2,
    # Strides for outputs [B, T, C]
    out_s0, out_s1, out_s2,
    # LayerNorm epsilon
    eps,
    # Compile-time block size (== C == 1024)
    BLOCK_C: tl.constexpr,
):
    """One program per (b, t) row.  Reads C elements, writes C*2 elements."""
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    c_offsets = tl.arange(0, BLOCK_C)   # [0, 1, ..., C-1]

    # ------------------------------------------------------------------ #
    # 1. Load tmp4[b, :, t]  (channels are the strided dimension)
    # ------------------------------------------------------------------ #
    tmp4_ptrs = tmp4_ptr + b * tmp4_s0 + c_offsets * tmp4_s1 + t * tmp4_s2
    x = tl.load(tmp4_ptrs)
    x_f32 = x.to(tl.float32)

    # ------------------------------------------------------------------ #
    # 2. Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # ------------------------------------------------------------------ #
    SQRT2_INV: tl.constexpr = 0.7071067811865476
    gelu = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * SQRT2_INV))

    # ------------------------------------------------------------------ #
    # 3. Load in3[b, t, :]  (contiguous over C)
    # ------------------------------------------------------------------ #
    in3_ptrs = in3_ptr + b * in3_s0 + t * in3_s1 + c_offsets * in3_s2
    y = tl.load(in3_ptrs)
    y_f32 = y.to(tl.float32)

    # ------------------------------------------------------------------ #
    # 4. Residual add  →  tmp_8  (dropout is identity in inference)
    # ------------------------------------------------------------------ #
    result = y_f32 + gelu   # shape [BLOCK_C]

    # Store tmp_8
    out1_ptrs = out1_ptr + b * out_s0 + t * out_s1 + c_offsets * out_s2
    tl.store(out1_ptrs, result.to(x.dtype))

    # ------------------------------------------------------------------ #
    # 5. LayerNorm over the C dimension
    # ------------------------------------------------------------------ #
    mean = tl.sum(result, axis=0) / C
    diff = result - mean
    var  = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm = diff * inv_std

    w    = tl.load(weight_ptr + c_offsets).to(tl.float32)
    b_ln = tl.load(bias_ptr   + c_offsets).to(tl.float32)
    out2 = norm * w + b_ln

    # Store tmp_10
    out2_ptrs = out2_ptr + b * out_s0 + t * out_s1 + c_offsets * out_s2
    tl.store(out2_ptrs, out2.to(x.dtype))


@torch.fx.wrap
def _fused_kernel_call_p005(tmp_4, in_3, in_1, in_0):
    """
    Inner kernel launcher decorated with @torch.fx.wrap so FX treats it as
    an opaque leaf node.  Returns a Python tuple (tmp_8, tmp_10).

    tmp_4 : [B, C, T]   – gelu input (non-contiguous slice of conv1d output)
    in_3  : [B, T, C]   – residual hidden states
    in_1  : [C]         – LayerNorm weight
    in_0  : [C]         – LayerNorm bias
    """
    B, C, T = tmp_4.shape

    out1 = torch.empty(B, T, C, dtype=tmp_4.dtype, device=tmp_4.device)
    out2 = torch.empty(B, T, C, dtype=tmp_4.dtype, device=tmp_4.device)

    grid = (B * T,)

    _fused_gelu_add_layernorm_kernel[grid](
        tmp_4, in_3, in_1, in_0,
        out1, out2,
        B, T, C,
        tmp_4.stride(0), tmp_4.stride(1), tmp_4.stride(2),
        in_3.stride(0),  in_3.stride(1),  in_3.stride(2),
        out1.stride(0),  out1.stride(1),  out1.stride(2),
        1e-5,
        BLOCK_C=1024,
    )

    return (out1, out2)


def fused_gelu_add_layernorm_p005(tmp_4, in_3, in_1, in_0):
    """
    Replacement callable traced by FX.  Calls the opaque kernel then uses
    getitem so the traced graph has two distinct returning nodes – matching
    the two returning nodes (tmp_8, tmp_10) of the matched pattern.
    """
    result = _fused_kernel_call_p005(tmp_4, in_3, in_1, in_0)
    return result[0], result[1]


# ------------------------------------------------------------------ #
# Pattern / replacement API expected by the AI4C framework
# ------------------------------------------------------------------ #

def pattern(tmp_4, in_3, in_1, in_0):
    tmp_5  = torch.nn.functional.gelu(tmp_4)
    tmp_6  = tmp_5.transpose(1, 2)
    tmp_7  = in_3 + tmp_6
    tmp_8  = torch.nn.functional.dropout(tmp_7, 0.05, False, False)
    tmp_10 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_10)


def replacement_args(tmp_4, in_3, in_1, in_0):
    return (tmp_4, in_3, in_1, in_0)


def replacement_func():
    return fused_gelu_add_layernorm_p005