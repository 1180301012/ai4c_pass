"""
Fuses the full WavLM GRU relative position computation for H=16 heads (wavlm_large).

Pattern:
  linear(in_3, in_1, in_0)            -> [1, 16, 199, 8]
  .view(1, 16, 199, 2, 4)             -> [1, 16, 199, 2, 4]
  .sum(-1)                            -> [1, 16, 199, 2]
  sigmoid                             -> [1, 16, 199, 2]
  chunk(2, dim=-1)                    -> 2 x [1, 16, 199, 1]
  chunk[1] * in_2 - 1.0              -> [1, 16, 199, 1]
  chunk[0] * (...) + 2.0             -> [1, 16, 199, 1]
  .view(1, 16, -1, 1)                -> [1, 16, 199, 1]

Each output element (h, p) depends only on in_3[0, h, p, :] (64 elements),
the full weight matrix in_1 [8, 64], bias in_0 [8], and scalar scale in_2[0, h, 0, 0].
We fuse everything into a single Triton kernel: one program per (head, position).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def fused_wavlm_large_kernel(
    in3_ptr,        # [1, H, N, K]  - input tensor
    in1_ptr,        # [8, K]        - weight matrix
    in0_ptr,        # [8]           - bias vector
    in2_ptr,        # [1, H, 1, 1]  - scale tensor
    out_ptr,        # [1, H, N, 1]  - output tensor
    stride_in3_h,   # stride of in_3 along head dim
    stride_in3_n,   # stride of in_3 along position dim
    stride_in2_h,   # stride of in_2 along head dim
    K: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
):
    """One Triton program per (head h, position p) pair."""
    pid = tl.program_id(0)
    h = pid // N
    p = pid % N

    k_offs = tl.arange(0, K)  # [K]

    # --- Load input vector x = in_3[0, h, p, :] ---
    x = tl.load(in3_ptr + h * stride_in3_h + p * stride_in3_n + k_offs)
    x_f32 = x.to(tl.float32)

    # --- Group 0: weight rows 0..3, compute dot products, sum, add biases ---
    j0_offs = tl.arange(0, 4)                                       # [4]
    W0 = tl.load(in1_ptr + j0_offs[:, None] * K + k_offs[None, :]) # [4, K]
    b0 = tl.load(in0_ptr + j0_offs)                                 # [4]
    dots0 = tl.sum(W0.to(tl.float32) * x_f32[None, :], axis=1)     # [4]
    s0 = tl.sum(dots0 + b0.to(tl.float32), axis=0)                  # scalar

    # --- Group 1: weight rows 4..7, compute dot products, sum, add biases ---
    j1_offs = 4 + tl.arange(0, 4)                                   # [4]
    W1 = tl.load(in1_ptr + j1_offs[:, None] * K + k_offs[None, :]) # [4, K]
    b1 = tl.load(in0_ptr + j1_offs)                                 # [4]
    dots1 = tl.sum(W1.to(tl.float32) * x_f32[None, :], axis=1)     # [4]
    s1 = tl.sum(dots1 + b1.to(tl.float32), axis=0)                  # scalar

    # --- Sigmoid ---
    g0 = tl.sigmoid(s0)
    g1 = tl.sigmoid(s1)

    # --- Load scale in_2[0, h, 0, 0] ---
    scale = tl.load(in2_ptr + h * stride_in2_h).to(tl.float32)

    # --- Compute final result: g0 * (g1 * scale - 1.0) + 2.0 ---
    result = g0 * (g1 * scale - 1.0) + 2.0

    # --- Store output[0, h, p, 0] ---
    tl.store(out_ptr + h * N + p, result.to(x.dtype))


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, 16, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 16, -1, 1)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_wavlm_large(in_0, in_1, in_2, in_3):
    H = 16
    N = 199
    K = 64
    out = torch.empty((1, H, N, 1), dtype=in_3.dtype, device=in_3.device)
    total = H * N  # 3184 programs
    fused_wavlm_large_kernel[(total,)](
        in_3, in_1, in_0, in_2, out,
        stride_in3_h=in_3.stride(1),
        stride_in3_n=in_3.stride(2),
        stride_in2_h=in_2.stride(1),
        K=K, H=H, N=N,
    )
    return out


def replacement_func():
    return fused_wavlm_large