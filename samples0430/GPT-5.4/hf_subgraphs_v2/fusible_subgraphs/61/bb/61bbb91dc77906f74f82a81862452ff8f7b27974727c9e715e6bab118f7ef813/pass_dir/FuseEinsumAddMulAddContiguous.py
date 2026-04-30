import operator
import torch
import triton
import triton.language as tl


# Pattern matching function
# NOTE: mirror model.py structure closely.
def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_5 = operator.iadd(in_3, einsum)
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_W': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_W': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['C', 'W', 'J'],
)
@triton.jit
def _fused_einsum_epilogue_tc_kernel(
    value_ptr,
    attn_ptr,
    bias_ptr,
    resid_ptr,
    scale_ptr,
    out_ptr,
    C,
    H,
    W,
    J,
    stride_vb,
    stride_vc,
    stride_vh,
    stride_vj,
    stride_ab,
    stride_ah,
    stride_aw,
    stride_aj,
    stride_bb,
    stride_bc,
    stride_bh,
    stride_bw,
    stride_rb,
    stride_rc,
    stride_rh,
    stride_rw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_c = tl.program_id(0)
    slice_id = tl.program_id(1)

    b = slice_id // H
    h = slice_id % H

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, BLOCK_W)
    c_mask = offs_c < C
    w_mask = offs_w < W

    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)

    for k_start in range(0, J, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < J

        a_ptrs = (
            value_ptr
            + b * stride_vb
            + offs_c[:, None] * stride_vc
            + h * stride_vh
            + offs_k[None, :] * stride_vj
        )
        b_ptrs = (
            attn_ptr
            + b * stride_ab
            + h * stride_ah
            + offs_w[:, None] * stride_aw
            + offs_k[None, :] * stride_aj
        )

        a = tl.load(a_ptrs, mask=c_mask[:, None] & k_mask[None, :], other=0.0)
        b_mat = tl.load(b_ptrs, mask=w_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(b_mat))

    out_ptrs = (
        out_ptr
        + b * stride_ob
        + offs_c[:, None] * stride_oc
        + h * stride_oh
        + offs_w[None, :] * stride_ow
    )
    resid_ptrs = (
        resid_ptr
        + b * stride_rb
        + offs_c[:, None] * stride_rc
        + h * stride_rh
        + offs_w[None, :] * stride_rw
    )
    bias_ptrs = (
        bias_ptr
        + b * stride_bb
        + offs_c[:, None] * stride_bc
        + h * stride_bh
        + offs_w[None, :] * stride_bw
    )
    out_mask = c_mask[:, None] & w_mask[None, :]

    resid = tl.load(resid_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    updated = resid + acc
    out = updated * scale + bias

    tl.store(resid_ptrs, updated, mask=out_mask)
    tl.store(out_ptrs, out, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_W': 64, 'BLOCK_K': 8}, num_warps=8, num_stages=2),
    ],
    key=['C', 'W', 'J'],
)
@triton.jit
def _fused_einsum_epilogue_fp32_kernel(
    value_ptr,
    attn_ptr,
    bias_ptr,
    resid_ptr,
    scale_ptr,
    out_ptr,
    C,
    H,
    W,
    J,
    stride_vb,
    stride_vc,
    stride_vh,
    stride_vj,
    stride_ab,
    stride_ah,
    stride_aw,
    stride_aj,
    stride_bb,
    stride_bc,
    stride_bh,
    stride_bw,
    stride_rb,
    stride_rc,
    stride_rh,
    stride_rw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_c = tl.program_id(0)
    slice_id = tl.program_id(1)

    b = slice_id // H
    h = slice_id % H

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, BLOCK_W)
    c_mask = offs_c < C
    w_mask = offs_w < W

    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)

    for k_start in range(0, J, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < J

        a_ptrs = (
            value_ptr
            + b * stride_vb
            + offs_c[:, None] * stride_vc
            + h * stride_vh
            + offs_k[None, :] * stride_vj
        )
        b_ptrs = (
            attn_ptr
            + b * stride_ab
            + h * stride_ah
            + offs_w[:, None] * stride_aw
            + offs_k[None, :] * stride_aj
        )

        a = tl.load(a_ptrs, mask=c_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        b_mat = tl.load(b_ptrs, mask=w_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        for kk in range(BLOCK_K):
            acc += a[:, kk][:, None] * b_mat[:, kk][None, :]

    out_ptrs = (
        out_ptr
        + b * stride_ob
        + offs_c[:, None] * stride_oc
        + h * stride_oh
        + offs_w[None, :] * stride_ow
    )
    resid_ptrs = (
        resid_ptr
        + b * stride_rb
        + offs_c[:, None] * stride_rc
        + h * stride_rh
        + offs_w[None, :] * stride_rw
    )
    bias_ptrs = (
        bias_ptr
        + b * stride_bb
        + offs_c[:, None] * stride_bc
        + h * stride_bh
        + offs_w[None, :] * stride_bw
    )
    out_mask = c_mask[:, None] & w_mask[None, :]

    resid = tl.load(resid_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    updated = resid + acc
    out = updated * scale + bias

    tl.store(resid_ptrs, updated, mask=out_mask)
    tl.store(out_ptrs, out, mask=out_mask)


@torch.fx.wrap
def fused_einsum_add_mul_add_contiguous(in_0, in_1, in_2, in_3, in_4):
    out = torch.empty_like(in_2)

    B, C, H, J = in_4.shape
    _, H2, W, J2 = in_1.shape
    if H != H2 or J != J2:
        # Keep a shape guard without using blocked torch APIs.
        raise RuntimeError('Input shapes do not satisfy expected einsum contract')

    grid = lambda META: (triton.cdiv(C, META['BLOCK_C']), B * H)

    if out.dtype == torch.float32:
        _fused_einsum_epilogue_fp32_kernel[grid](
            in_4,
            in_1,
            in_2,
            in_3,
            in_0,
            out,
            C,
            H,
            W,
            J,
            in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
    else:
        _fused_einsum_epilogue_tc_kernel[grid](
            in_4,
            in_1,
            in_2,
            in_3,
            in_0,
            out,
            C,
            H,
            W,
            J,
            in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_einsum_add_mul_add_contiguous