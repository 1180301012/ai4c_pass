"""
Fused RMSNorm kernel for SmolLM3 variant:
  - epsilon = 1e-6
  - input in_2: bfloat16
  - weight in_0: bfloat16
  - output: bfloat16

Fuses: in_2.to(f32) -> pow(2) -> mean(-1,keepdim) -> +eps -> rsqrt -> mul -> to(bf16) -> mul(weight)
"""

import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@triton.jit
def _rms_norm_eps1e6_bf16_kernel(
    X_ptr,   # [N, H] bfloat16 input
    W_ptr,   # [H]    bfloat16 weight
    Y_ptr,   # [N, H] bfloat16 output
    H,       # hidden dim
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)

    # Load input as float32 (no mask needed when BLOCK_H == H)
    x = tl.load(X_ptr + row * H + offsets, eviction_policy="evict_first").to(tl.float32)

    # Compute inverse RMS: rsqrt(mean(x^2) + eps)
    x2_sum = tl.sum(x * x, axis=0)
    rrms = tl.rsqrt(x2_sum / H + 1e-6)

    # Load weight as float32 – keep in cache (small, reused across all rows)
    w = tl.load(W_ptr + offsets, eviction_policy="evict_last").to(tl.float32)

    # Normalize and scale, store as bfloat16
    tl.store(Y_ptr + row * H + offsets, (x * rrms * w).to(tl.bfloat16))


@torch.fx.wrap
def rms_norm_eps1e6_bf16_wrapper(in_0, in_2):
    """
    in_0: [H] bfloat16 – layernorm weight
    in_2: [*batch, H] bfloat16 – input embeddings
    returns: [*batch, H] bfloat16
    """
    x = in_2.contiguous()
    w = in_0.contiguous()

    orig_shape = x.shape
    H = orig_shape[-1]
    N = x.numel() // H

    # Large N: fused single-pass Triton kernel
    x_flat = x.view(N, H)
    y_flat = torch.empty((N, H), dtype=torch.bfloat16, device=x.device)

    _rms_norm_eps1e6_bf16_kernel[(N,)](
        x_flat, w, y_flat,
        H,
        BLOCK_H=2048,
        num_warps=8,
    )

    return y_flat.view(orig_shape)


def replacement_func():
    return rms_norm_eps1e6_bf16_wrapper