import torch
import triton
import triton.language as tl


def pattern(in_0, softmax_out):
    tmp_1 = in_0 * softmax_out
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, softmax_out):
    return (in_0, softmax_out)


@triton.jit
def _mul_sum_dim1_2way_contig_kernel(
    x_ptr,
    w_ptr,
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

    w_base = n * (2 * C) + c
    w0 = tl.load(w_ptr + w_base).to(tl.float32)
    w1 = tl.load(w_ptr + w_base + C).to(tl.float32)

    x0 = tl.load(x_ptr + x0_base + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x1_base + offs, mask=mask, other=0.0).to(tl.float32)
    out = x0 * w0 + x1 * w1
    tl.store(out_ptr + out_base + offs, out, mask=mask)


@triton.jit
def _mul_sum_dim1_2way_strided_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    C,
    W,
    S,
    sx_n,
    sx_b,
    sx_c,
    sx_h,
    sx_w,
    sw_n,
    sw_b,
    sw_c,
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

    w_base = n * sw_n + c * sw_c
    w0 = tl.load(w_ptr + w_base).to(tl.float32)
    w1 = tl.load(w_ptr + w_base + sw_b).to(tl.float32)

    x0_offs = n * sx_n + c * sx_c + h * sx_h + w * sx_w
    x1_offs = x0_offs + sx_b
    x0 = tl.load(x_ptr + x0_offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x1_offs, mask=mask, other=0.0).to(tl.float32)
    out = x0 * w0 + x1 * w1

    out_offs = n * so_n + c * so_c + h * so_h + w * so_w
    tl.store(out_ptr + out_offs, out, mask=mask)


@torch.fx.wrap
def fused_mul_sum_dim1_2way(in_0, softmax_out):
    n = in_0.shape[0]
    c = in_0.shape[2]
    h = in_0.shape[3]
    w = in_0.shape[4]
    s = h * w

    out = torch.empty((n, c, h, w), device=in_0.device, dtype=in_0.dtype)

    if in_0.is_contiguous() and softmax_out.is_contiguous() and out.is_contiguous():
        cs = c * s
        bcs = 2 * cs
        grid = (n * c,)
        if s <= 256:
            _mul_sum_dim1_2way_contig_kernel[grid](
                in_0, softmax_out, out, c, s, cs, bcs, BLOCK_S=256, num_warps=1, num_stages=2
            )
        elif s <= 512:
            _mul_sum_dim1_2way_contig_kernel[grid](
                in_0, softmax_out, out, c, s, cs, bcs, BLOCK_S=512, num_warps=2, num_stages=2
            )
        else:
            _mul_sum_dim1_2way_contig_kernel[grid](
                in_0, softmax_out, out, c, s, cs, bcs, BLOCK_S=1024, num_warps=4, num_stages=2
            )
        return out

    grid = lambda meta: (n * c, triton.cdiv(s, meta["BLOCK_S"]))
    _mul_sum_dim1_2way_strided_kernel[grid](
        in_0,
        softmax_out,
        out,
        c,
        w,
        s,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        in_0.stride(4),
        softmax_out.stride(0),
        softmax_out.stride(1),
        softmax_out.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_S=256,
        num_warps=2,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_mul_sum_dim1_2way