import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim = 2)
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = tmp_5.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit

def _distance_scale_softmax_kernel(
    x_ptr,
    code_ptr,
    scale_ptr,
    out_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_k,
    code_stride_c,
    code_stride_k,
    scale_stride_c,
    out_stride_n,
    out_stride_c,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    acc0 = tl.zeros((8,), dtype=tl.float32)
    acc1 = tl.zeros((8,), dtype=tl.float32)
    acc2 = tl.zeros((8,), dtype=tl.float32)
    acc3 = tl.zeros((8,), dtype=tl.float32)

    offs_c0 = tl.arange(0, 8)
    offs_c1 = 8 + tl.arange(0, 8)
    offs_c2 = 16 + tl.arange(0, 8)
    offs_c3 = 24 + tl.arange(0, 8)

    for k0 in tl.static_range(0, 512, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < 512

        x0_ptrs = x_ptr + pid_n * x_stride_n + offs_c0[:, None] * x_stride_c + offs_k[None, :] * x_stride_k
        w0_ptrs = code_ptr + offs_c0[:, None] * code_stride_c + offs_k[None, :] * code_stride_k
        x0 = tl.load(x0_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        w0 = tl.load(w0_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        d0 = x0 - w0
        acc0 += tl.sum(d0 * d0, axis=1)

        x1_ptrs = x_ptr + pid_n * x_stride_n + offs_c1[:, None] * x_stride_c + offs_k[None, :] * x_stride_k
        w1_ptrs = code_ptr + offs_c1[:, None] * code_stride_c + offs_k[None, :] * code_stride_k
        x1 = tl.load(x1_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        w1 = tl.load(w1_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        d1 = x1 - w1
        acc1 += tl.sum(d1 * d1, axis=1)

        x2_ptrs = x_ptr + pid_n * x_stride_n + offs_c2[:, None] * x_stride_c + offs_k[None, :] * x_stride_k
        w2_ptrs = code_ptr + offs_c2[:, None] * code_stride_c + offs_k[None, :] * code_stride_k
        x2 = tl.load(x2_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        d2 = x2 - w2
        acc2 += tl.sum(d2 * d2, axis=1)

        x3_ptrs = x_ptr + pid_n * x_stride_n + offs_c3[:, None] * x_stride_c + offs_k[None, :] * x_stride_k
        w3_ptrs = code_ptr + offs_c3[:, None] * code_stride_c + offs_k[None, :] * code_stride_k
        x3 = tl.load(x3_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        w3 = tl.load(w3_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        d3 = x3 - w3
        acc3 += tl.sum(d3 * d3, axis=1)

    s0 = tl.load(scale_ptr + offs_c0 * scale_stride_c).to(tl.float32)
    s1 = tl.load(scale_ptr + offs_c1 * scale_stride_c).to(tl.float32)
    s2 = tl.load(scale_ptr + offs_c2 * scale_stride_c).to(tl.float32)
    s3 = tl.load(scale_ptr + offs_c3 * scale_stride_c).to(tl.float32)

    l0 = acc0 * s0
    l1 = acc1 * s1
    l2 = acc2 * s2
    l3 = acc3 * s3

    m0 = tl.max(l0, axis=0)
    m1 = tl.max(l1, axis=0)
    m2 = tl.max(l2, axis=0)
    m3 = tl.max(l3, axis=0)
    m = tl.maximum(tl.maximum(m0, m1), tl.maximum(m2, m3))

    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    e2 = tl.exp(l2 - m)
    e3 = tl.exp(l3 - m)
    denom = tl.sum(e0, axis=0) + tl.sum(e1, axis=0) + tl.sum(e2, axis=0) + tl.sum(e3, axis=0)

    p0 = e0 / denom
    p1 = e1 / denom
    p2 = e2 / denom
    p3 = e3 / denom

    tl.store(out_ptr + pid_n * out_stride_n + offs_c0 * out_stride_c, p0)
    tl.store(out_ptr + pid_n * out_stride_n + offs_c1 * out_stride_c, p1)
    tl.store(out_ptr + pid_n * out_stride_n + offs_c2 * out_stride_c, p2)
    tl.store(out_ptr + pid_n * out_stride_n + offs_c3 * out_stride_c, p3)


@triton.jit

def _broadcast_sub_kernel(
    x_ptr,
    code_ptr,
    out_ptr,
    x_stride_n,
    x_stride_k,
    code_stride_c,
    code_stride_k,
    out_stride_n,
    out_stride_c,
    out_stride_k,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_cb = tl.program_id(1)
    pid_kb = tl.program_id(2)

    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_k = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_c = offs_c < 32
    mask_k = offs_k < 512
    mask = mask_c[:, None] & mask_k[None, :]

    x = tl.load(x_ptr + pid_n * x_stride_n + offs_k * x_stride_k, mask=mask_k, other=0.0)
    code = tl.load(code_ptr + offs_c[:, None] * code_stride_c + offs_k[None, :] * code_stride_k, mask=mask, other=0.0)
    out = x[None, :] - code

    out_ptrs = out_ptr + pid_n * out_stride_n + offs_c[:, None] * out_stride_c + offs_k[None, :] * out_stride_k
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def encnet_encoding_fullgraph_triton(in_0, in_1, in_2, in_3, in_4):
    n = in_4.shape[1]
    k = in_4.shape[2]

    out_delta = torch.empty((1, n, 32, k), device=in_4.device, dtype=in_4.dtype)
    out_soft = torch.empty((1, n, 32, 1), device=in_1.device, dtype=in_1.dtype)

    _distance_scale_softmax_kernel[(n,)](
        in_1,
        in_2,
        in_3,
        out_soft,
        in_1.stride()[1],
        in_1.stride()[2],
        in_1.stride()[3],
        in_2.stride()[2],
        in_2.stride()[3],
        in_3.stride()[2],
        out_soft.stride()[1],
        out_soft.stride()[2],
        BLOCK_K=128,
        num_warps=4,
        num_stages=4,
    )

    grid = (n, triton.cdiv(32, 8), triton.cdiv(k, 128))
    _broadcast_sub_kernel[grid](
        in_4,
        in_0,
        out_delta,
        in_4.stride()[1],
        in_4.stride()[2],
        in_0.stride()[0],
        in_0.stride()[1],
        out_delta.stride()[1],
        out_delta.stride()[2],
        out_delta.stride()[3],
        BLOCK_C=8,
        BLOCK_K=128,
        num_warps=8,
        num_stages=4,
    )

    return (out_delta, out_soft)



def replacement_func():
    return encnet_encoding_fullgraph_triton