"""
Full-graph fusion for SmolLM3 variant:
  - RoPE: cat(x,x,dim=-1) -> cos -> *1.0 -> to(bf16) and sin -> *1.0 -> to(bf16)
  - RMSNorm: to(f32) -> pow(2) -> mean(-1,keepdim) -> +1e-6 -> rsqrt -> mul -> to(bf16) -> mul(weight)
  - eps = 1e-6, RoPE and RMSNorm outputs are bfloat16

Matches: bfloat16/0, float16/4, bfloat16/8, float32/5, float16/7, float32/7, float32/9, float16/9, float32/0, float16/0
"""

import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    # ---- RoPE branch ----
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    # ---- RMSNorm branch ----
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_6, tmp_17, tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# RMSNorm Triton kernel  (bfloat16 input, bfloat16 output)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 2048}, num_warps=8),
        triton.Config({"BLOCK_H": 2048}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _rmsnorm_bf16_kernel(
    X_ptr, W_ptr, Y_ptr,
    H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < H

    x = tl.load(X_ptr + row * H + offsets, mask=mask, other=0.0).to(tl.float32)
    x2_sum = tl.sum(x * x, axis=0)
    rrms = tl.rsqrt(x2_sum / H + 1e-6)
    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    y = (x * rrms * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * H + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# RoPE cos/sin Triton kernel  (any-dtype input, bfloat16 output)
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bf16_kernel(
    X_ptr, COS_ptr, SIN_ptr,
    N, K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(X_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    c = tl.cos(x).to(tl.bfloat16)
    s = tl.sin(x).to(tl.bfloat16)

    # Write both halves of the doubled output
    tl.store(COS_ptr + row * 2 * K + offsets,     c, mask=mask)
    tl.store(SIN_ptr + row * 2 * K + offsets,     s, mask=mask)
    tl.store(COS_ptr + row * 2 * K + K + offsets, c, mask=mask)
    tl.store(SIN_ptr + row * 2 * K + K + offsets, s, mask=mask)


@torch.fx.wrap
def fuse_full_graph_smollm3(in_0, in_1, in_2):
    """
    in_0: [H]          bfloat16  – layernorm weight
    in_1: [*, K]       any dtype – frequency (RoPE) tensor
    in_2: [*, H]       bfloat16  – input embeddings
    returns: (cos_bf16 [*,2K], rms_out_bf16 [*,H], sin_bf16 [*,2K])
    """
    # ---- RMSNorm ----
    x2 = in_2.contiguous()
    w  = in_0.contiguous()
    shape2 = x2.shape
    H = shape2[-1]
    N2 = x2.numel() // H
    x2_flat = x2.view(N2, H)
    rms_flat = torch.empty((N2, H), dtype=torch.bfloat16, device=x2.device)
    _rmsnorm_bf16_kernel[(N2,)](x2_flat, w, rms_flat, H)
    rms_out = rms_flat.view(shape2)

    # ---- RoPE ----
    x1 = in_1.contiguous()
    shape1 = x1.shape
    K = shape1[-1]
    N1 = x1.numel() // K
    BLOCK_K = triton.next_power_of_2(K)

    out_shape = list(shape1)
    out_shape[-1] = 2 * K
    cos_out = torch.empty(out_shape, dtype=torch.bfloat16, device=x1.device)
    sin_out = torch.empty(out_shape, dtype=torch.bfloat16, device=x1.device)

    x1_flat   = x1.view(N1, K)
    cos_flat  = cos_out.view(N1, 2 * K)
    sin_flat  = sin_out.view(N1, 2 * K)

    _rope_bf16_kernel[(N1,)](
        x1_flat, cos_flat, sin_flat,
        N1, K,
        BLOCK_K=BLOCK_K,
    )

    return cos_out, rms_out, sin_out


def replacement_func():
    return fuse_full_graph_smollm3