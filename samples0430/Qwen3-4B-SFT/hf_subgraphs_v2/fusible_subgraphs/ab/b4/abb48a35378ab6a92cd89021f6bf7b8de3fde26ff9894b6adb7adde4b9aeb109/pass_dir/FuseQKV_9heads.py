"""
Fused QKV projection pass for N=9 heads (convit_small).
Fuses: linear → reshape(1,197,3,9,48) → permute(2,0,3,1,4)
       → unbind(0) → getitem[0]/[1]/[2] → transpose(-2,-1) on K
into a single Triton GEMM + scatter kernel.
"""
import torch
import triton
import triton.language as tl

from pass_dir.qkv_kernel import fused_qkv_forward  # noqa: F401


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Wrapper (must be @torch.fx.wrap) ─────────────────────────────────────────

@torch.fx.wrap
def triton_qkv_n9(in_0, in_1):
    # in_0: weight [3*9*48, K] = [1296, K]
    # in_1: input  [1, 197, K]
    return fused_qkv_forward(in_0, in_1, N=9, HEAD_DIM=48)


def replacement_func():
    return triton_qkv_n9