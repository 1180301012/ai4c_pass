import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 8}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def fused_post_conv_kernel(
    conv_ptr,
    in2_ptr,
    in3_ptr,
    in4_ptr,
    out_ptr,
    conv_s0, conv_s1, conv_s2, conv_s3,
    in2_s0, in2_s1, in2_s2, in2_s3,
    in3_s0, in3_s1, in3_s2, in3_s3,
    in4_s0, in4_s1, in4_s2, in4_s3,
    out_s0, out_s1, out_s2, out_s3,
    N,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = offs_h[:, None]
    ow = offs_w[None, :]

    mask = (pid_nc < N * C) & (oh < 64) & (ow < 64)

    ph = oh % 4
    pw = ow % 4

    h0_raw = tl.where(oh >= 2, (oh - 2) // 4, 0)
    w0_raw = tl.where(ow >= 2, (ow - 2) // 4, 0)
    h0 = tl.minimum(h0_raw, 15)
    w0 = tl.minimum(w0_raw, 15)
    h1 = tl.minimum(h0 + 1, 15)
    w1 = tl.minimum(w0 + 1, 15)

    lh = tl.where(
        oh < 2,
        0.0,
        tl.where(
            ph == 0,
            0.625,
            tl.where(ph == 1, 0.875, tl.where(ph == 2, 0.125, 0.375)),
        ),
    )
    lw = tl.where(
        ow < 2,
        0.0,
        tl.where(
            pw == 0,
            0.625,
            tl.where(pw == 1, 0.875, tl.where(pw == 2, 0.125, 0.375)),
        ),
    )

    one_minus_lh = 1.0 - lh
    one_minus_lw = 1.0 - lw

    conv_base = n * conv_s0 + c * conv_s1
    in2_base = n * in2_s0 + c * in2_s1
    in3_base = n * in3_s0 + c * in3_s1
    in4_base = n * in4_s0 + c * in4_s1
    out_base = n * out_s0 + c * out_s1

    out_ptrs = out_ptr + out_base + oh * out_s2 + ow * out_s3
    in3_ptrs = in3_ptr + in3_base + oh * in3_s2 + ow * in3_s3

    x3 = tl.load(in3_ptrs, mask=mask, other=0.0).to(tl.float32)

    in4_ptr_00 = in4_ptr + in4_base + h0 * in4_s2 + w0 * in4_s3
    in4_ptr_01 = in4_ptr + in4_base + h0 * in4_s2 + w1 * in4_s3
    in4_ptr_10 = in4_ptr + in4_base + h1 * in4_s2 + w0 * in4_s3
    in4_ptr_11 = in4_ptr + in4_base + h1 * in4_s2 + w1 * in4_s3

    a00 = tl.load(in4_ptr_00, mask=mask, other=0.0).to(tl.float32)
    a01 = tl.load(in4_ptr_01, mask=mask, other=0.0).to(tl.float32)
    a10 = tl.load(in4_ptr_10, mask=mask, other=0.0).to(tl.float32)
    a11 = tl.load(in4_ptr_11, mask=mask, other=0.0).to(tl.float32)

    interp_a = (
        a00 * one_minus_lh * one_minus_lw
        + a01 * one_minus_lh * lw
        + a10 * lh * one_minus_lw
        + a11 * lh * lw
    )
    sig_a = 1.0 / (1.0 + tl.exp(-interp_a))
    branch_a = x3 * sig_a

    conv_ptr_00 = conv_ptr + conv_base + h0 * conv_s2 + w0 * conv_s3
    conv_ptr_01 = conv_ptr + conv_base + h0 * conv_s2 + w1 * conv_s3
    conv_ptr_10 = conv_ptr + conv_base + h1 * conv_s2 + w0 * conv_s3
    conv_ptr_11 = conv_ptr + conv_base + h1 * conv_s2 + w1 * conv_s3

    in2_ptr_00 = in2_ptr + in2_base + h0 * in2_s2 + w0 * in2_s3
    in2_ptr_01 = in2_ptr + in2_base + h0 * in2_s2 + w1 * in2_s3
    in2_ptr_10 = in2_ptr + in2_base + h1 * in2_s2 + w0 * in2_s3
    in2_ptr_11 = in2_ptr + in2_base + h1 * in2_s2 + w1 * in2_s3

    b00_conv = tl.load(conv_ptr_00, mask=mask, other=0.0).to(tl.float32)
    b01_conv = tl.load(conv_ptr_01, mask=mask, other=0.0).to(tl.float32)
    b10_conv = tl.load(conv_ptr_10, mask=mask, other=0.0).to(tl.float32)
    b11_conv = tl.load(conv_ptr_11, mask=mask, other=0.0).to(tl.float32)

    b00_in2 = tl.load(in2_ptr_00, mask=mask, other=0.0).to(tl.float32)
    b01_in2 = tl.load(in2_ptr_01, mask=mask, other=0.0).to(tl.float32)
    b10_in2 = tl.load(in2_ptr_10, mask=mask, other=0.0).to(tl.float32)
    b11_in2 = tl.load(in2_ptr_11, mask=mask, other=0.0).to(tl.float32)

    p00 = b00_in2 * (1.0 / (1.0 + tl.exp(-b00_conv)))
    p01 = b01_in2 * (1.0 / (1.0 + tl.exp(-b01_conv)))
    p10 = b10_in2 * (1.0 / (1.0 + tl.exp(-b10_conv)))
    p11 = b11_in2 * (1.0 / (1.0 + tl.exp(-b11_conv)))

    branch_b = (
        p00 * one_minus_lh * one_minus_lw
        + p01 * one_minus_lh * lw
        + p10 * lh * one_minus_lw
        + p11 * lh * lw
    )

    out = branch_a + branch_b
    tl.store(out_ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N_ELEMS'],
)
@triton.jit
def sigmoid_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_ELEMS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMS
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = y * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(out_ptr + offsets, out, mask=mask)


def _run_full(conv_out, in_2, in_3, in_4):
    n = conv_out.shape[0]
    c = conv_out.shape[1]
    out = torch.empty_like(in_3)
    grid = lambda meta: (
        triton.cdiv(64, meta['BLOCK_W']),
        triton.cdiv(64, meta['BLOCK_H']),
        n * c,
    )
    fused_post_conv_kernel[grid](
        conv_out,
        in_2,
        in_3,
        in_4,
        out,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n,
        c,
    )
    return out


def _run_sigmul(conv_out, in_2):
    out = torch.empty_like(in_2)
    n_elems = in_2.numel()
    grid = lambda meta: (triton.cdiv(n_elems, meta['BLOCK_SIZE']),)
    sigmoid_mul_kernel[grid](conv_out, in_2, out, n_elems)
    return out


def _run_branch_a(in_3, in_4):
    # Reuse the full fused kernel by feeding zeros into branch B.
    conv_out = torch.zeros_like(in_4)
    in_2 = torch.zeros_like(in_4)
    return _run_full(conv_out, in_2, in_3, in_4)



def shared_dispatch(*args):
    route = args[-1]
    if route == 'full':
        return _run_full(args[0], args[1], args[2], args[3])
    if route == 'sigmul':
        return _run_sigmul(args[0], args[1])
    if route == 'branch_a':
        return _run_branch_a(args[0], args[1])
    raise RuntimeError(f'Unknown route: {route}')