import torch
import triton
import triton.language as tl


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
def _einsum_tc_kernel(
    value_ptr,
    attn_ptr,
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
    out_mask = c_mask[:, None] & w_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_W': 64, 'BLOCK_K': 8}, num_warps=8, num_stages=2),
    ],
    key=['C', 'W', 'J'],
)
@triton.jit
def _einsum_fp32_kernel(
    value_ptr,
    attn_ptr,
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
    out_mask = c_mask[:, None] & w_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _pointwise_epilogue_kernel(
    x_ptr,
    bias_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    out = x * scale + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def ccnet_dispatch(*args):
    route = args[-1]

    if route == 'einsum':
        value = args[0]
        attn = args[1]
        B, C, H, J = value.shape
        _, H2, W, J2 = attn.shape
        if H != H2 or J != J2:
            raise RuntimeError('Input shapes do not satisfy expected einsum contract')

        out = torch.empty((B, C, H, W), device=value.device, dtype=value.dtype)
        grid = lambda META: (triton.cdiv(C, META['BLOCK_C']), B * H)

        _einsum_tc_kernel[grid](
            value,
            attn,
            out,
            C,
            H,
            W,
            J,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out

    if route == 'epilogue':
        x = args[0]
        scale = args[1]
        bias = args[2]
        out = torch.empty_like(x)
        n_elements = out.numel()
        grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
        _pointwise_epilogue_kernel[grid](
            x,
            bias,
            scale,
            out,
            n_elements,
        )
        return out

    raise RuntimeError('Unknown route')


def shared_replacement_func():
    return ccnet_dispatch