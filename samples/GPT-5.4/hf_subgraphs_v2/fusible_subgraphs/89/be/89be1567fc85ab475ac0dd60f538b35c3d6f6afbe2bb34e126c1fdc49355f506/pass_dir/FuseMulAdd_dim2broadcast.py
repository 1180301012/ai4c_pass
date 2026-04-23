import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor
from pass_dir.fuse_mul_add_helper import fused_mul_add_compiled


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_mul_add_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    in0_s0,
    in0_s1,
    in1_s0,
    in1_s1,
    in1_s2,
    in1_s3,
    in2_s0,
    in2_s1,
    in2_s2,
    in2_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    n_rows,
    T,
    C,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)

    row_ids = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = row_ids < n_rows
    b = row_ids // T
    t = row_ids % T

    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    mask = row_mask[:, None] & c_mask[None, :]

    # Load tiny parameter tensors once per program and broadcast across BLOCK_R rows
    w0_ptrs = in1_ptr + c * in1_s3
    w1_ptrs = in1_ptr + in1_s2 + c * in1_s3
    w0 = tl.load(w0_ptrs, mask=c_mask, other=0.0)[None, :]
    w1 = tl.load(w1_ptrs, mask=c_mask, other=0.0)[None, :]

    beta0_ptrs = in0_ptr + c * in0_s1
    beta1_ptrs = in0_ptr + in0_s0 + c * in0_s1
    beta0 = tl.load(beta0_ptrs, mask=c_mask, other=0.0)[None, :]
    beta1 = tl.load(beta1_ptrs, mask=c_mask, other=0.0)[None, :]

    x_ptrs = in2_ptr + b[:, None] * in2_s0 + t[:, None] * in2_s1 + c[None, :] * in2_s3
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    y0 = x * w0 + beta0
    y1 = x * w1 + beta1

    out0_ptrs = out_ptr + b[:, None] * out_s0 + t[:, None] * out_s1 + c[None, :] * out_s3
    out1_ptrs = out_ptr + b[:, None] * out_s0 + t[:, None] * out_s1 + out_s2 + c[None, :] * out_s3
    tl.store(out0_ptrs, y0, mask=mask)
    tl.store(out1_ptrs, y1, mask=mask)


@torch.fx.wrap
def fused_mul_add(in_0, in_1, in_2):
    in_0 = unwrap_tensor(in_0)
    in_1 = unwrap_tensor(in_1)
    in_2 = unwrap_tensor(in_2)
    return fused_mul_add_compiled(in_0, in_1, in_2)


def replacement_func():
    return fused_mul_add