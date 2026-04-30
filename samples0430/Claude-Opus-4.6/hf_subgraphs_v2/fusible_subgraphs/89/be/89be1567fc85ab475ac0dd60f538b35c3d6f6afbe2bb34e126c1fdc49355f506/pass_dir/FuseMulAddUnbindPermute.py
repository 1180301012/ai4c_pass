import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_mul_add_unbind_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_0_ptr,
    out_1_ptr,
    K: tl.constexpr,
):
    # One program per (b, s) pair
    pid = tl.program_id(0)
    offs = tl.arange(0, K)
    base = pid * K

    # Load in_2[pid, :]
    x = tl.load(in_2_ptr + base + offs)

    # Load scale and bias vectors
    a0 = tl.load(in_1_ptr + offs)
    a1 = tl.load(in_1_ptr + K + offs)
    b0 = tl.load(in_0_ptr + offs)
    b1 = tl.load(in_0_ptr + K + offs)

    # Fused multiply-add for both output slices
    tl.store(out_0_ptr + base + offs, x * a0 + b0)
    tl.store(out_1_ptr + base + offs, x * a1 + b1)


@torch.fx.wrap
def fused_mul_add_unbind(in_0, in_1, in_2):
    B = in_2.shape[0]
    S = in_2.shape[1]
    K = in_2.shape[3]

    out_0 = torch.empty((B, S, K), dtype=in_2.dtype, device=in_2.device)
    out_1 = torch.empty((B, S, K), dtype=in_2.dtype, device=in_2.device)

    fused_mul_add_unbind_kernel[(B * S,)](
        in_0, in_1, in_2,
        out_0, out_1,
        K=K,
        num_warps=1,
        num_stages=1,
    )

    return (out_0, out_1)


def replacement_func():
    return fused_mul_add_unbind