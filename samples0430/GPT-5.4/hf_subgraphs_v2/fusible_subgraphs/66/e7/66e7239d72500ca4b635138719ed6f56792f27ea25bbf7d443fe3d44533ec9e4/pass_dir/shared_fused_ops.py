import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_768_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    B,
    S,
    H,
    x_s0,
    x_s1,
    x_s2,
    out_s0,
    out_s1,
    out_s2,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    rows = B * S
    if row >= rows:
        return

    b = row // S
    s = row % S
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    x = tl.load(x_ptr + b * x_s0 + s * x_s1 + offs * x_s2, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / H
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / H
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_centered * rstd * gamma + beta

    tl.store(out_ptr + b * out_s0 + s * out_s1 + offs * out_s2, y, mask=mask)


@triton.jit
def expand_cast_mask_kernel(
    mask_ptr,
    out_ptr,
    mask_s0,
    mask_s1,
    out_s0,
    out_s1,
    out_s2,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= 16:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask_cols = offs < 768
    m = tl.load(mask_ptr + row * mask_s1).to(tl.float32)
    tl.store(out_ptr + row * out_s1 + offs * out_s2, m, mask=mask_cols)


@triton.jit
def mul_f32_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, a * b, mask=mask)


@torch.fx.wrap
def shared_runtime_dispatch(in_0, in_1=None, in_2=None, in_3=None, route=None):
    if route == "layer_norm":
        x = in_3
        gamma = in_2
        beta = in_1
        out = torch.empty_like(x)
        rows = x.shape[0] * x.shape[1]
        layer_norm_768_kernel[(rows,)](
            x,
            gamma,
            beta,
            out,
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            1e-12,
            BLOCK_SIZE=1024,
            num_warps=4,
            num_stages=1,
        )
        return out

    if route == "mask_expand_float":
        ref = in_3
        mask = in_0
        out = torch.empty_like(ref, dtype=torch.float32)
        expand_cast_mask_kernel[(16,)](
            mask,
            out,
            mask.stride(0),
            mask.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=1,
        )
        return out

    if route == "mul":
        a = in_3
        b = in_0
        out = torch.empty_like(a, dtype=torch.float32)
        n = a.numel()
        mul_f32_kernel[((n + 1023) // 1024,)](
            a,
            b,
            out,
            n,
            BLOCK_SIZE=1024,
            num_warps=4,
            num_stages=1,
        )
        return out

    return in_0


def shared_replacement_func():
    return shared_runtime_dispatch