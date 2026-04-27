"""
Shared Triton kernel and route-dispatch wrapper.

Key design:
- D is a runtime arg with do_not_specialize → ONE compiled kernel for all D values
- BLOCK_D=1024 fixed constexpr → single compilation per pointer dtype
- Two kernels total: fp16-in/fp16-out and bf16-in/bf16-out
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["D"])
def add_ln_kernel_fp16(
    a_ptr, b_ptr, w_ptr, bias_ptr, out_ptr,
    D,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    a = tl.load(a_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    mean = tl.sum(x, axis=0) / D
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)
    var  = tl.sum(diff * diff, axis=0) / D
    inv_std = tl.rsqrt(var + eps)
    x_norm  = diff * inv_std

    w    = tl.load(w_ptr    + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out  = w * x_norm + bias

    tl.store(out_ptr + row * D + offs, out.to(tl.float16), mask=mask)


@triton.jit(do_not_specialize=["D"])
def add_ln_kernel_bf16(
    a_ptr, b_ptr, w_ptr, bias_ptr, out_ptr,
    D,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    a = tl.load(a_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    mean = tl.sum(x, axis=0) / D
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)
    var  = tl.sum(diff * diff, axis=0) / D
    inv_std = tl.rsqrt(var + eps)
    x_norm  = diff * inv_std

    w    = tl.load(w_ptr    + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out  = w * x_norm + bias

    tl.store(out_ptr + row * D + offs, out.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def pass_dispatch(in_0, emb, weight, bias, route):
    """
    Single @torch.fx.wrap entry-point shared by all add+LN passes.
    D and rows are read from tensor shapes at runtime.
    """
    shape = in_0.shape
    rows  = shape[0] * shape[1]
    D     = shape[2]
    out   = torch.empty_like(in_0)
    if in_0.dtype == torch.bfloat16:
        add_ln_kernel_bf16[(rows,)](
            in_0, emb, weight, bias, out,
            D=D, BLOCK_D=1024, eps=1e-5,
            num_warps=4, num_stages=1,
        )
    else:
        add_ln_kernel_fp16[(rows,)](
            in_0, emb, weight, bias, out,
            D=D, BLOCK_D=1024, eps=1e-5,
            num_warps=4, num_stages=1,
        )
    return out