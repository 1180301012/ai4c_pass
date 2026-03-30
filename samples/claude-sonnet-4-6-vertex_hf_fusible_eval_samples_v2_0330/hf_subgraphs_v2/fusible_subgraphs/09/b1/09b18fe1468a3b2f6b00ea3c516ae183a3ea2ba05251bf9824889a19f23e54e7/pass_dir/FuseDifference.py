"""
FuseDifference: Fuses in_0.view() + in_4.unsqueeze(2).expand() - in_0.view()

Input shapes:
  in_0: [32, 512]        (codewords)
  in_4: [1, 4096, 512]   (x_7 feature map)

Output shape:
  [1, 4096, 32, 512]     (tmp_10 = pixel-codeword differences)

Strategy:
  Grid [N*K] = [4096*32]. Each program handles one (n, k) pair and
  computes the D=512-element difference: out[n, k, :] = in_4[n, :] - in_0[k, :]
  in native dtype (float16 or bfloat16) without materializing the expand.

  Memory savings: avoids writing 128 MB for the expanded intermediate tmp_8.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern (mirrors model.py exactly)
# ---------------------------------------------------------------------------

def pattern(in_0, in_4):
    tmp_6  = in_0.view((1, 1, 32, 512))
    tmp_7  = in_4.unsqueeze(2)
    tmp_8  = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


# ---------------------------------------------------------------------------
# Kernel: broadcast difference, grid [N*K]
# Each program handles one (n, k) pair, processes BLOCK_D features.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["N", "K", "D"],
)
@triton.jit
def difference_kernel(
    in0_ptr, in4_ptr, out_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
):
    pid    = tl.program_id(0)   # pid = n * K + k
    n      = pid // K
    k      = pid % K

    d_offs = tl.arange(0, BLOCK_D)

    # Load in_4[n, :] and in_0[k, :] in native precision
    x4 = tl.load(in4_ptr + n * D + d_offs)
    x0 = tl.load(in0_ptr + k * D + d_offs)

    diff = x4 - x0                              # native precision

    tl.store(out_ptr + pid * D + d_offs, diff)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def compute_difference(in_0, in_4):
    # in_0: [32, 512]
    # in_4: [1, 4096, 512]
    K = in_0.shape[0]    # 32
    D = in_0.shape[1]    # 512
    B = in_4.shape[0]    # 1
    N = in_4.shape[1]    # 4096

    out = torch.empty(N * K * D, dtype=in_0.dtype, device=in_0.device)

    difference_kernel[(N * K,)](
        in_0, in_4, out,
        N, K, D,
    )

    # Reshape to [B, N, K, D] = [1, 4096, 32, 512]
    return out.view(B, N, K, D)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_4):
    return (in_0, in_4)


def replacement_func():
    return compute_difference