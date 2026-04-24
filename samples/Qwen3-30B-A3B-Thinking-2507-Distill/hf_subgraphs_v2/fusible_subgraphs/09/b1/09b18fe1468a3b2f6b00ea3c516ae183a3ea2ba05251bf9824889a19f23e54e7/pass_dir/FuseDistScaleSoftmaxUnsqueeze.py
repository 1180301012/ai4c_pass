"""
Fuses Path 1 of EncNet_R101_start361_end371_4:
  (in_1 - in_2).pow(2).sum(dim=3) * in_3  ->  tmp_4 [1,4096,32]

Inputs:
  in_1: [1, 4096, 32, 512]   (expanded_x)
  in_2: [1, 1,    32, 512]   (reshaped_codewords)
  in_3: [1, 1,    32]        (reshaped_scale)

Output:
  tmp_4: [1, 4096, 32]  (scaled distance logits, consumed by softmax)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 8,  'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 8,  'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 16, 'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 16, 'BLOCK_D': 512}, num_warps=16),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 512}, num_warps=16),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 512}, num_warps=32),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 512}, num_warps=16),
    ],
    key=['N', 'C', 'D'],
)
@triton.jit
def fused_dist_scale_kernel(
    in1_ptr,   # [N, C, D]  (B=1)  in1[n, c, d] = in1_ptr + n*C*D + c*D + d
    in2_ptr,   # [C, D]               in2[c, d] = in2_ptr + c*D + d
    in3_ptr,   # [C]                  in3[c]    = in3_ptr + c
    out_ptr,   # [N, C]               out[n, c] = out_ptr + n*C + c
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,  # number of n-values per program
    BLOCK_D: tl.constexpr,  # number of d-values per program (= D)
):
    # 2D grid: pid_n in [0, N/BLOCK_N), pid_c in [0, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_offs = tl.arange(0, BLOCK_D)                      # [BLOCK_D]
    mask_n = n_offs < N                                  # [BLOCK_N]

    # Load scale scalar for this channel
    scale_val = tl.load(in3_ptr + pid_c).to(tl.float32)

    # 2D load: in1[n_offs, pid_c, d_offs]
    base1 = n_offs[:, None] * C * BLOCK_D + pid_c * BLOCK_D + d_offs[None, :]
    x1 = tl.load(in1_ptr + base1, mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_D]

    # 1D load: in2[pid_c, d_offs] — shared across n-programs with same pid_c
    x2 = tl.load(in2_ptr + pid_c * BLOCK_D + d_offs).to(tl.float32)               # [BLOCK_D]

    # Compute per-n sum of squared differences, then scale
    diff = x1 - x2[None, :]
    ss = tl.sum(diff * diff, axis=1)                                                # [BLOCK_N]
    out_vals = scale_val * ss

    # Store: out[n_offs, pid_c]
    tl.store(out_ptr + n_offs * C + pid_c, out_vals.to(in1_ptr.dtype.element_ty), mask=mask_n)


@torch.fx.wrap
def fused_dist_scale(in_1, in_2, in_3):
    # in_1: [1, N, C, D]
    # in_2: [1, 1, C, D]
    # in_3: [1, 1, C]
    B, N, C, D = in_1.shape
    out = torch.empty((B, N, C), dtype=in_1.dtype, device=in_1.device)

    # Autotune finds best BLOCK_N / num_warps; grid is a lambda
    def grid(meta):
        return ((N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'], C)

    fused_dist_scale_kernel[grid](
        in_1, in_2, in_3, out,
        N, C, D,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


def replacement_func():
    return fused_dist_scale