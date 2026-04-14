import torch
import triton
import triton.language as tl


# ── GEMV kernel ───────────────────────────────────────────────────────────────
# 512 programs, one per output feature.
# Each program:
#   - loads the full 512-element input vector x (hits L1/L2 after first SM)
#   - loads its weight row w[pid, :] from global memory
#   - computes the dot product + bias and stores the scalar result
# Uses tl.sum(..., axis=0) which returns a 0-d scalar, allowing the scalar store
# tl.store(out_ptr + pid, scalar).  K=512 as tl.constexpr here does NOT cause
# register spilling because there is no unrolled K-loop; the whole vector is
# loaded in one shot and reduced in one tl.sum call.
@triton.jit
def linear_gemv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    k_offs = tl.arange(0, K)
    x = tl.load(x_ptr + k_offs).to(tl.float32)
    w = tl.load(w_ptr + pid * K + k_offs).to(tl.float32)
    b = tl.load(b_ptr + pid).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x * w, axis=0) + b)


@torch.fx.wrap
def fused_linear_view_transpose_contiguous(x, weight, bias):
    # x:      [1, 1, 512]
    # weight: [512, 512]
    # bias:   [512]
    # output: [1, 8, 1, 64]
    # Allocate directly as [1,8,1,64] (same flat layout as [512] for seq_len=1)
    # so no reshape is needed after the kernel writes sequentially.
    out = torch.empty(1, 8, 1, 64, dtype=x.dtype, device=x.device)
    linear_gemv_kernel[(512,)](x, weight, bias, out, K=512)
    return out


def pattern(x, weight, bias):
    linear = torch.nn.functional.linear(x, weight, bias)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_10


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return fused_linear_view_transpose_contiguous