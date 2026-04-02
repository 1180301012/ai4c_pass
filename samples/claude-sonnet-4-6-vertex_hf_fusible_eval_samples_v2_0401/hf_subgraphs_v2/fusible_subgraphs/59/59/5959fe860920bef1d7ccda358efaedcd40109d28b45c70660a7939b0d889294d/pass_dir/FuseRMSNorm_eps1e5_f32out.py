"""
Fused RMSNorm kernel for TinyLlama variant:
  - epsilon = 1e-5
  - input in_2: bfloat16
  - weight in_0: bfloat16
  - output: float32  (bf16 weight * f32 normed = f32)

Fuses: in_2.to(f32) -> pow(2) -> mean(-1,keepdim) -> +eps -> rsqrt -> mul -> to(f32) -> mul(weight)
"""

import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-05
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.float32)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@triton.jit
def _rms_norm_eps1e5_f32_kernel(
    X_ptr,   # [N, H] bfloat16 input
    W_ptr,   # [H]    bfloat16 weight
    Y_ptr,   # [N, H] float32  output
    H,       # hidden dim
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)

    # Load input as float32
    x = tl.load(X_ptr + row * H + offsets).to(tl.float32)

    # Compute inverse RMS: rsqrt(mean(x^2) + eps)
    x2_sum = tl.sum(x * x, axis=0)
    rrms = tl.rsqrt(x2_sum / H + 1e-5)

    # Load weight as float32 (original is bfloat16, promote for mul)
    w = tl.load(W_ptr + offsets).to(tl.float32)

    # Normalize and scale, store as float32
    tl.store(Y_ptr + row * H + offsets, x * rrms * w)


@torch.fx.wrap
def rms_norm_eps1e5_f32_wrapper(in_0, in_2):
    """
    in_0: [H] bfloat16 – layernorm weight
    in_2: [*batch, H] bfloat16 – input embeddings
    returns: [*batch, H] float32
    """
    x = in_2.contiguous()
    w = in_0.contiguous()

    orig_shape = x.shape
    H = orig_shape[-1]
    N = x.numel() // H

    x_flat = x.view(N, H)
    y_flat = torch.empty((N, H), dtype=torch.float32, device=x.device)

    _rms_norm_eps1e5_f32_kernel[(N,)](
        x_flat, w, y_flat,
        H,
        BLOCK_H=2048,
        num_warps=16,
    )

    return y_flat.view(orig_shape)


def replacement_func():
    return rms_norm_eps1e5_f32_wrapper