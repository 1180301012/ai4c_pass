"""
FuseSoftAssign: Fuses softmax(x, dim=2).unsqueeze(3) → tmp_9

Pattern matches the final two ops of the soft-assignment chain:
    tmp_5 = softmax(tmp_4, dim=2)   [1, 4096, 32]
    tmp_9 = tmp_5.unsqueeze(3)      [1, 4096, 32, 1]

Replacement: single fast Triton softmax kernel (numerically stable),
Grid[N=4096], one pixel per program over K=32 codewords.
Output is produced directly in the correct dtype.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern — matches softmax + unsqueeze subgraph
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_5 = torch.nn.functional.softmax(x, dim = 2)
    tmp_9 = tmp_5.unsqueeze(3)
    return tmp_9


# ---------------------------------------------------------------------------
# Kernel: stable softmax over K=32, Grid[N=4096]
#   x_ptr: [1, N, K]  — x[0, n, k] at offset n*K+k
#   out_ptr: [1, N, K, 1] — same flat layout
# ---------------------------------------------------------------------------

@triton.jit
def softmax_unsqueeze_kernel(
    x_ptr, out_ptr,
    N,
    K_: tl.constexpr,
):
    n      = tl.program_id(0)
    k_offs = tl.arange(0, K_)

    xv     = tl.load(x_ptr + n * K_ + k_offs).to(tl.float32)

    xv_max = tl.max(xv, axis=0)
    xv_exp = tl.exp(xv - xv_max)
    soft   = xv_exp / tl.sum(xv_exp, axis=0)     # fp32

    tl.store(out_ptr + n * K_ + k_offs, soft)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def soft_assign(x):
    # x: [1, 4096, 32]  (tmp_4 = in_3 * scaled_distances)
    N = x.shape[1]    # 4096
    K = x.shape[2]    # 32

    out_f32 = torch.empty(N * K, dtype=torch.float32, device=x.device)

    softmax_unsqueeze_kernel[(N,)](
        x, out_f32,
        N, K_=32,
        num_warps=4,
    )

    # [1, N, K, 1]
    return out_f32.to(x.dtype).view(1, N, K, 1)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def replacement_args(x):
    return (x,)


def replacement_func():
    return soft_assign