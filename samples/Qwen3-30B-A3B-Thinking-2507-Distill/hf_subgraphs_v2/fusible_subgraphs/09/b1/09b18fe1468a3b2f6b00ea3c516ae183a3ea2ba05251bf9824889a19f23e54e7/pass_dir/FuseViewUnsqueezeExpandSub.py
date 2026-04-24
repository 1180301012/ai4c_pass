"""
Fuses Path 2 of EncNet_R101_start361_end371_4:
  in_0.view(1,1,32,512) + in_4.unsqueeze(2).expand(1,4096,32,512) -> sub  -> tmp_10

Inputs:
  in_0: [32, 512]        (codewords)
  in_4: [1, 4096, 512]   (x_7)

Output:
  tmp_10: [1, 4096, 32, 512]

For B=1:
  tmp_10[0, n, c, m] = in_4[0, n, m] - in_0[c, m]

Grid: (N * C,) programs, each handles M=512 elements for one (n, c) pair.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def view_unsqueeze_expand_sub_kernel(
    in4_ptr,   # [1, N, M]  (B=1)
    in0_ptr,   # [C, M]
    out_ptr,   # [1, N, C, M]  -> [N, C, M] effectively
    N,
    C: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # One program per (n, c) pair  (B=1)
    nc = tl.program_id(0)
    n = nc // C
    c = nc % C

    # Load in4[0, n, :] and in0[c, :] in tiles of BLOCK_M
    for m_start in range(0, M, BLOCK_M):
        m_range = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_range < M

        base4 = n * M + m_range
        base0 = c * M + m_range

        x4 = tl.load(in4_ptr + base4, mask=mask_m, other=0.0)
        x0 = tl.load(in0_ptr + base0, mask=mask_m, other=0.0)

        diff = x4 - x0
        # out[0, n, c, m] = out_ptr + n*C*M + c*M + m = nc*M + m
        tl.store(out_ptr + nc * M + m_range, diff, mask=mask_m)


@torch.fx.wrap
def fused_view_unsqueeze_expand_sub(in_0, in_4):
    # in_0: [C, M]
    # in_4: [1, N, M]
    C, M = in_0.shape
    B, N, _ = in_4.shape
    out = torch.empty((B, N, C, M), dtype=in_4.dtype, device=in_4.device)

    view_unsqueeze_expand_sub_kernel[(B * N * C,)](
        in_4, in_0, out,
        N, C, M,
        BLOCK_M=512,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    return (in_0, in_4)


def replacement_func():
    return fused_view_unsqueeze_expand_sub