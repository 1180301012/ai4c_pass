import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_mul_add_unbind_permute_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_perm_ptr,
    out_first_ptr,
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
    outp_s0,
    outp_s1,
    outp_s2,
    outf_s0,
    outf_s1,
    outf_s2,
    T,
    C,
    BLOCK_C: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_c = tl.program_id(1)

    b = pid_row // T
    t = pid_row % T

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x_ptrs = in2_ptr + b * in2_s0 + t * in2_s1 + offs_c * in2_s3
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    w0_ptrs = in1_ptr + offs_c * in1_s3
    w1_ptrs = in1_ptr + in1_s2 + offs_c * in1_s3
    w0 = tl.load(w0_ptrs, mask=mask, other=0.0)
    w1 = tl.load(w1_ptrs, mask=mask, other=0.0)

    b0_ptrs = in0_ptr + offs_c * in0_s1
    b1_ptrs = in0_ptr + in0_s0 + offs_c * in0_s1
    b0 = tl.load(b0_ptrs, mask=mask, other=0.0)
    b1 = tl.load(b1_ptrs, mask=mask, other=0.0)

    y0 = x * w0 + b0
    y1 = x * w1 + b1

    out_first_ptrs = out_first_ptr + b * outf_s0 + t * outf_s1 + offs_c * outf_s2
    out_perm_ptrs = out_perm_ptr + b * outp_s0 + offs_c * outp_s1 + t * outp_s2

    tl.store(out_first_ptrs, y0, mask=mask)
    tl.store(out_perm_ptrs, y1, mask=mask)


@torch.fx.wrap
def fused_mul_add_unbind_permute(in_0, in_1, in_2):
    in_0 = unwrap_tensor(in_0)
    in_1 = unwrap_tensor(in_1)
    in_2 = unwrap_tensor(in_2)

    B = in_2.shape[0]
    T = in_2.shape[1]
    C = in_0.shape[1]

    out_perm = torch.empty((B, C, T), device=in_2.device, dtype=in_2.dtype)
    out_first = torch.empty((B, T, C), device=in_2.device, dtype=in_2.dtype)

    rows = B * T
    if rows <= 64:
        block_c = 32
        num_warps = 1
    elif rows <= 256:
        block_c = 64
        num_warps = 2
    else:
        block_c = 128
        num_warps = 4

    grid = (rows, (C + block_c - 1) // block_c)

    in0_s0, in0_s1 = in_0.stride()
    in1_s0, in1_s1, in1_s2, in1_s3 = in_1.stride()
    in2_s0, in2_s1, in2_s2, in2_s3 = in_2.stride()
    outp_s0, outp_s1, outp_s2 = out_perm.stride()
    outf_s0, outf_s1, outf_s2 = out_first.stride()

    fused_mul_add_unbind_permute_kernel[grid](
        in_0,
        in_1,
        in_2,
        out_perm,
        out_first,
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
        outp_s0,
        outp_s1,
        outp_s2,
        outf_s0,
        outf_s1,
        outf_s2,
        T,
        C,
        BLOCK_C=block_c,
        num_warps=num_warps,
        num_stages=2,
    )

    return (out_perm, out_first)


def replacement_func():
    return fused_mul_add_unbind_permute