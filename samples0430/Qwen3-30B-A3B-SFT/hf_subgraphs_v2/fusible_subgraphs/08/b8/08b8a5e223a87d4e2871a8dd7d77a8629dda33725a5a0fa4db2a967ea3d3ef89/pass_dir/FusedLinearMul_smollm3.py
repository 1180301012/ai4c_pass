"""
Pass: FusedLinearMul_smollm3

Matches the pattern (SmolLM3 / Gemma):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return (tmp_2,)

in_0 : weight   [N, K]       e.g. [11008, 2048]  or [16384, 2048]
in_1 : input    [..., K]    e.g. [64, 128, 2048] or [1, 3, 2048]
in_2 : multiply factor [..., N] e.g. [64, 128, 11008] or [1, 3, 16384]

Strategy:
  - Single fused Triton kernel: computes out = (in_1 @ in_0.T) * in_2
    where in_2 is treated as a broadcast scale over all leading dims.
  - Uses the shared fused_linear_scale_mul kernel.
"""

import torch
import triton
import triton.language as tl

from pass_dir.fused_linear_scale_mul import fused_linear_scale_mul_kernel


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = in_2 * linear
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_linear_mul_smollm3(in_0, in_1, in_2):
    """
    in_0 : weight   [N, K]
    in_1 : input    [..., K]
    in_2 : scale/mul factor [..., N]  -- multiplied channel-wise

    Returns (in_2 * (in_1 @ in_0.T),)

    The shared fused_linear_scale_mul_kernel(x=in_1, w=in_0, scale=in_2)
    computes (in_1 @ in_0.T) * in_2 directly.
    For contiguous in_2 with last-dim == N, the flat channel index for
    element at flat offset i is i % N, which equals the channel index of
    the corresponding GEMM output.  The kernel loads scale[c] using the
    N index, so this is correct.
    """
    K = in_1.shape[-1]
    N = in_0.shape[0]
    M = in_1.numel() // K

    in1_2d = in_1.reshape(M, K)
    in0_2d = in_0.reshape(N, K)

    # in_2 is [..., N]; treat it as the scale vector of length N
    out_2d = torch.empty((M, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))
    fused_linear_scale_mul_kernel[grid](
        in1_2d, in0_2d, in_2, out_2d,
        M, N, K,
        in1_2d.stride(0), in1_2d.stride(1),
        in0_2d.stride(0), in0_2d.stride(1),
    )

    # Restore original leading shape: in_2 has one more dim than in_1
    # (e.g. in_1 is [B, S, K], in_2 is [B, S, N])
    return (out_2d.reshape(*in_2.shape[:-1], N),)


def replacement_func():
    return fused_linear_mul_smollm3