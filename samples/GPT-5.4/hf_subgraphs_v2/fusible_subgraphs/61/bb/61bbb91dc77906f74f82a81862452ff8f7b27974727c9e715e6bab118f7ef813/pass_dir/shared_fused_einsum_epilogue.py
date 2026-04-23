import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["C", "W", "J"],
)
@triton.jit
def _fused_einsum_epilogue_kernel(
    gamma_ptr,
    attn_ptr,
    bias_ptr,
    residual_ptr,
    value_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    J,
    stride_attn_b,
    stride_attn_h,
    stride_attn_w,
    stride_attn_j,
    stride_bias_b,
    stride_bias_c,
    stride_bias_h,
    stride_bias_w,
    stride_res_b,
    stride_res_c,
    stride_res_h,
    stride_res_w,
    stride_val_b,
    stride_val_c,
    stride_val_h,
    stride_val_j,
    stride_out_b,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, J, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = (
            value_ptr
            + b * stride_val_b
            + offs_m[:, None] * stride_val_c
            + h * stride_val_h
            + offs_k[None, :] * stride_val_j
        )
        b_ptrs = (
            attn_ptr
            + b * stride_attn_b
            + h * stride_attn_h
            + offs_n[:, None] * stride_attn_w
            + offs_k[None, :] * stride_attn_j
        )

        a_mask = (offs_m[:, None] < C) & (offs_k[None, :] < J)
        b_mask = (offs_n[:, None] < W) & (offs_k[None, :] < J)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, tl.trans(b_tile))

    mask = (offs_m[:, None] < C) & (offs_n[None, :] < W)
    res_ptrs = (
        residual_ptr
        + b * stride_res_b
        + offs_m[:, None] * stride_res_c
        + h * stride_res_h
        + offs_n[None, :] * stride_res_w
    )
    bias_ptrs = (
        bias_ptr
        + b * stride_bias_b
        + offs_m[:, None] * stride_bias_c
        + h * stride_bias_h
        + offs_n[None, :] * stride_bias_w
    )
    out_ptrs = (
        out_ptr
        + b * stride_out_b
        + offs_m[:, None] * stride_out_c
        + h * stride_out_h
        + offs_n[None, :] * stride_out_w
    )

    residual = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr).to(tl.float32)
    out = (acc + residual) * gamma + bias
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def fused_einsum_residual_mul_add_contiguous_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    B = in_4.shape[0]
    C = in_4.shape[1]
    H = in_4.shape[2]
    J = in_4.shape[3]
    W = in_1.shape[2]

    out = torch.empty_like(in_2)

    _ = route
    grid = lambda META: (triton.cdiv(C, META["BLOCK_M"]), B * H)
    _fused_einsum_epilogue_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        in_4,
        out,
        B,
        C,
        H,
        W,
        J,
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        in_4.stride(0),
        in_4.stride(1),
        in_4.stride(2),
        in_4.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return (out,)


def replacement_func():
    return fused_einsum_residual_mul_add_contiguous_dispatch