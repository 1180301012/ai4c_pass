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
def _fused_softmax_mul_sum_dim1_2way_contig_kernel(
    x_ptr,
    logits_ptr,
    out_ptr,
    C,
    S,
    CS,
    BCS,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    n = pid // C

    offs = tl.arange(0, BLOCK_S)
    mask = offs < S

    x0_base = n * BCS + c * S
    x1_base = x0_base + CS
    out_base = n * CS + c * S

    logits_base = n * (2 * C) + c
    l0 = tl.load(logits_ptr + logits_base).to(tl.float32)
    l1 = tl.load(logits_ptr + logits_base + C).to(tl.float32)
    p1 = 1.0 / (1.0 + tl.exp(l0 - l1))

    x0 = tl.load(x_ptr + x0_base + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x1_base + offs, mask=mask, other=0.0).to(tl.float32)
    out = x0 + (x1 - x0) * p1
    tl.store(out_ptr + out_base + offs, out, mask=mask)


@triton.jit
def _fused_softmax_mul_sum_dim1_2way_strided_kernel(
    x_ptr,
    logits_ptr,
    out_ptr,
    C,
    W,
    S,
    sx_n,
    sx_b,
    sx_c,
    sx_h,
    sx_w,
    sl_n,
    sl_b,
    sl_c,
    so_n,
    so_c,
    so_h,
    so_w,
    BLOCK_S: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs_s < S
    h = offs_s // W
    w = offs_s - h * W

    logits_base = n * sl_n + c * sl_c
    l0 = tl.load(logits_ptr + logits_base).to(tl.float32)
    l1 = tl.load(logits_ptr + logits_base + sl_b).to(tl.float32)
    p1 = 1.0 / (1.0 + tl.exp(l0 - l1))

    x0_offs = n * sx_n + c * sx_c + h * sx_h + w * sx_w
    x1_offs = x0_offs + sx_b
    x0 = tl.load(x_ptr + x0_offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x1_offs, mask=mask, other=0.0).to(tl.float32)
    out = x0 + (x1 - x0) * p1

    out_offs = n * so_n + c * so_c + h * so_h + w * so_w
    tl.store(out_ptr + out_offs, out, mask=mask)


@torch.fx.wrap
def fused_softmax_mul_sum_dim1_2way(in_0, in_1):
    tmp_0 = in_1.softmax(1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = tmp_1.sum(1)
    return tmp_2


def replacement_func():
    return fused_softmax_mul_sum_dim1_2way