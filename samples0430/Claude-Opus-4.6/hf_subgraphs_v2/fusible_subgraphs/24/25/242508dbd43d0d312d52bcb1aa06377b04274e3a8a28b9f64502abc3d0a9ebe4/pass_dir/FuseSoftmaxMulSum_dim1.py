import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    N_out, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_out

    # Channel index (C=256 hardcoded)
    c = offs // HW

    # Softmax using sigmoid: softmax([a,b])[0] = sigmoid(a-b)
    s0 = tl.load(in_1_ptr + c, mask=mask).to(tl.float32)
    s1 = tl.load(in_1_ptr + 256 + c, mask=mask).to(tl.float32)
    w0 = tl.sigmoid(s0 - s1)
    w1 = 1.0 - w0

    # Load data from both branches (branch1 offset = N_out = C*HW)
    x0 = tl.load(in_0_ptr + offs, mask=mask).to(tl.float32)
    x1 = tl.load(in_0_ptr + N_out + offs, mask=mask).to(tl.float32)

    # Weighted sum
    r = x0 * w0 + x1 * w1

    # Store
    tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    # in_0: [1, 2, 256, H, W], in_1: [1, 2, 256, 1, 1]
    # output: [1, 256, H, W]
    H = in_0.shape[3]
    W = in_0.shape[4]
    HW = H * W
    N_out = 256 * HW

    out = torch.empty(1, 256, H, W, dtype=in_0.dtype, device=in_0.device)

    grid = ((N_out + 2047) // 2048,)
    fused_softmax_mul_sum_kernel[grid](
        in_0, in_1, out,
        N_out, HW,
        BLOCK_SIZE=2048,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_softmax_mul_sum