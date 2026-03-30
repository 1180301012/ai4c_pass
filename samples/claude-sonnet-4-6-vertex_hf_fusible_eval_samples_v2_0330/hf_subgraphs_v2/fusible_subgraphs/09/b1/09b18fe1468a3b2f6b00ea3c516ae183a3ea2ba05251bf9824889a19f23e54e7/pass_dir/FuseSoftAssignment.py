"""
FuseSoftAssignment: Fuses (in_1 - in_2).pow(2).sum(dim=3) * in_3 -> softmax(dim=2) -> unsqueeze(3)

Input shapes:
  in_1: [1, 4096, 32, 512]  (expanded_x)
  in_2: [1,    1, 32, 512]  (reshaped_codewords)
  in_3: [1,    1, 32]       (reshaped_scale)

Output shape:
  [1, 4096, 32, 1]           (tmp_9 = soft assignments)

Strategy:
  Kernel A (distance_kernel): Grid [N*K] = [4096*32].
      Each program computes scale[k] * sum_d((in_1[n,k,d] - in_2[k,d])^2) -> float32 intermediate.
  Kernel B (softmax_kernel): Grid [N] = [4096].
      Each program loads K=32 distances, applies numerically stable softmax, stores float32.
  Python wrapper converts float32 result to input dtype and reshapes to [1,N,K,1].

This eliminates three 128 MB intermediate tensor materializations (tmp_1, tmp_2, tmp_3).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern (mirrors model.py exactly)
# ---------------------------------------------------------------------------

def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim = 2)
    tmp_9 = tmp_5.unsqueeze(3)
    return tmp_9


# ---------------------------------------------------------------------------
# Kernel A: squared-distance + scale, grid [N*K]
# Each program handles one (n, k) pair, reduces over D=512.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_warps=4),
    ],
    key=["N", "K", "D"],
)
@triton.jit
def distance_kernel(
    in1_ptr, in2_ptr, in3_ptr, dist_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
):
    pid  = tl.program_id(0)       # pid = n * K + k
    n    = pid // K
    k    = pid % K

    d_offs = tl.arange(0, BLOCK_D)

    # Load in_1[n, k, :] and in_2[k, :], accumulate squared diff in fp32
    x = tl.load(in1_ptr + n * K * D + k * D + d_offs).to(tl.float32)
    c = tl.load(in2_ptr +             k * D + d_offs).to(tl.float32)

    diff  = x - c
    total = tl.sum(diff * diff, axis=0)          # scalar float32

    # Scale by in_3[k]
    scale = tl.load(in3_ptr + k).to(tl.float32)
    total = total * scale

    tl.store(dist_ptr + pid, total)


# ---------------------------------------------------------------------------
# Kernel B: numerically-stable softmax over K=32, grid [N]
# ---------------------------------------------------------------------------

@triton.jit
def softmax_kernel(
    dist_ptr, out_ptr,
    N,
    K_: tl.constexpr,
):
    n      = tl.program_id(0)
    k_offs = tl.arange(0, K_)

    dist = tl.load(dist_ptr + n * K_ + k_offs)   # float32, shape [K_]

    # Numerically stable softmax
    d_max  = tl.max(dist, axis=0)
    d_exp  = tl.exp(dist - d_max)
    d_sum  = tl.sum(d_exp, axis=0)
    soft   = d_exp / d_sum                        # float32

    tl.store(out_ptr + n * K_ + k_offs, soft)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def soft_assignment(in_1, in_2, in_3):
    B = in_1.shape[0]   # 1
    N = in_1.shape[1]   # 4096
    K = in_1.shape[2]   # 32
    D = in_1.shape[3]   # 512

    dtype  = in_1.dtype
    device = in_1.device

    # Intermediate float32 buffers
    dist   = torch.empty(N * K,     dtype=torch.float32, device=device)
    out_f32 = torch.empty(N * K,    dtype=torch.float32, device=device)

    # Kernel A: compute scaled squared distances
    distance_kernel[(N * K,)](
        in_1, in_2, in_3, dist,
        N, K, D,
    )

    # Kernel B: softmax over K
    softmax_kernel[(N,)](
        dist, out_f32,
        N, K_=32,
    )

    # Convert to input dtype, reshape to [B, N, K, 1]
    return out_f32.to(dtype).view(B, N, K, 1)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


def replacement_func():
    return soft_assignment