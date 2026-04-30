import torch
import triton
import triton.language as tl


@triton.jit
def _l2norm_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    x_stride_last,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x_raw = tl.load(x_ptr + offs * x_stride_last, mask=mask, other=0)
    x = x_raw.to(tl.float32)
    denom = tl.sqrt(tl.sum(x * x, axis=0))
    normed = x / denom
    tl.store(out_ptr + offs, normed.to(x_raw.dtype), mask=mask)


@triton.jit
def _exp_mul_kernel(
    scale_ptr,
    x_ptr,
    out_ptr,
    n_cols,
    x_stride_last,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x_raw = tl.load(x_ptr + offs * x_stride_last, mask=mask, other=0)
    x = x_raw.to(tl.float32)
    scale = tl.exp(tl.load(scale_ptr + 0).to(tl.float32))
    out = x * scale
    tl.store(out_ptr + offs, out.to(x_raw.dtype), mask=mask)


@torch.fx.wrap
def xclip_dispatch(arg0, arg1, route):
    if route == "norm_only":
        x = arg0
        n_cols = x.shape[-1]
        out = torch.empty_like(x)
        _l2norm_kernel[(1,)](
            x,
            out,
            n_cols,
            x.stride(-1),
            BLOCK_SIZE=512,
            num_warps=4,
            num_stages=1,
        )
        return out
    elif route == "exp_mul":
        scale_in = arg0
        x = arg1
        n_cols = x.shape[-1]
        out = torch.empty_like(x)
        _exp_mul_kernel[(1,)](
            scale_in,
            x,
            out,
            n_cols,
            x.stride(-1),
            BLOCK_SIZE=512,
            num_warps=4,
            num_stages=1,
        )
        return out
    return arg0


def replacement_func():
    return xclip_dispatch