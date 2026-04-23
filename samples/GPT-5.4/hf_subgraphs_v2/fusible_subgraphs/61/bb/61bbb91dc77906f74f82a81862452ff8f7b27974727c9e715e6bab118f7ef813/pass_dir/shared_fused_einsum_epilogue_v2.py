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
def _einsum_bchj_bhwj_to_bchw_kernel(
    attn_ptr,
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

        value_ptrs = (
            value_ptr
            + b * stride_val_b
            + offs_m[:, None] * stride_val_c
            + h * stride_val_h
            + offs_k[None, :] * stride_val_j
        )
        attn_ptrs = (
            attn_ptr
            + b * stride_attn_b
            + h * stride_attn_h
            + offs_n[:, None] * stride_attn_w
            + offs_k[None, :] * stride_attn_j
        )

        value_mask = (offs_m[:, None] < C) & (offs_k[None, :] < J)
        attn_mask = (offs_n[:, None] < W) & (offs_k[None, :] < J)

        value = tl.load(value_ptrs, mask=value_mask, other=0.0)
        attn = tl.load(attn_ptrs, mask=attn_mask, other=0.0)
        acc += tl.dot(value, tl.trans(attn))

    out_ptrs = (
        out_ptr
        + b * stride_out_b
        + offs_m[:, None] * stride_out_c
        + h * stride_out_h
        + offs_n[None, :] * stride_out_w
    )
    out_mask = (offs_m[:, None] < C) & (offs_n[None, :] < W)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _mul_add_kernel(
    x_ptr,
    gamma_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr).to(tl.float32)
    out = x * gamma + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]

    if route == "route_einsum":
        value = args[0]
        attn = args[1]

        B = value.shape[0]
        C = value.shape[1]
        H = value.shape[2]
        J = value.shape[3]
        W = attn.shape[2]

        out = torch.empty((B, C, H, W), device=value.device, dtype=value.dtype)
        grid = lambda META: (triton.cdiv(C, META["BLOCK_M"]), B * H)
        _einsum_bchj_bhwj_to_bchw_kernel[grid](
            attn,
            value,
            out,
            B,
            C,
            H,
            W,
            J,
            attn.stride(0),
            attn.stride(1),
            attn.stride(2),
            attn.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
        return out

    if route == "route_pointwise_tuple":
        x = args[0]
        gamma = args[1]
        bias = args[2]

        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _mul_add_kernel[grid](x, gamma, bias, out, n_elements)
        return (out,)

    if route == "route_pointwise_tensor":
        x = args[0]
        gamma = args[1]
        bias = args[2]

        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _mul_add_kernel[grid](x, gamma, bias, out, n_elements)
        return out

    x = args[0]
    gamma = args[1]
    bias = args[2]
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _mul_add_kernel[grid](x, gamma, bias, out, n_elements)
    return (out,)


def replacement_func():
    return shared_dispatch