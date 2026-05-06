import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: layer-norm with normalised_shape=(192,)  [route "192"]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-05)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "192")


# ---------------------------------------------------------------------------
# Triton layer-norm kernel – BLOCK_C=512 for C=384 (placeholder branch)
# ---------------------------------------------------------------------------

@triton.jit
def _ln_kernel_384(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C, eps,
    BLOCK_C: tl.constexpr,
):
    row  = tl.program_id(0)
    off  = tl.arange(0, BLOCK_C)
    mask = off < C
    base = row * C
    xf   = tl.load(x_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(xf, axis=0) / C
    diff = tl.where(mask, xf - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w    = tl.load(w_ptr + off, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + off, (diff * rstd * w + b).to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ln_kernel_192(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C, eps,
    BLOCK_C: tl.constexpr,
):
    row  = tl.program_id(0)
    off  = tl.arange(0, BLOCK_C)
    mask = off < C
    base = row * C
    xf   = tl.load(x_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(xf, axis=0) / C
    diff = tl.where(mask, xf - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w    = tl.load(w_ptr + off, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + off, (diff * rstd * w + b).to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ln_kernel_96(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C, eps,
    BLOCK_C: tl.constexpr,
):
    row  = tl.program_id(0)
    off  = tl.arange(0, BLOCK_C)
    mask = off < C
    base = row * C
    xf   = tl.load(x_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(xf, axis=0) / C
    diff = tl.where(mask, xf - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w    = tl.load(w_ptr + off, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + off, (diff * rstd * w + b).to(x_ptr.dtype.element_ty), mask=mask)


def _run_kernel(weight, bias, x, route):
    C   = x.shape[-1]
    N   = x.numel() // C
    out = torch.empty_like(x)
    if route == "384":
        _ln_kernel_384[(N,)](x, weight, bias, out, C, 1e-5,
                             BLOCK_C=512, num_warps=8)
    elif route == "192":
        _ln_kernel_192[(N,)](x, weight, bias, out, C, 1e-5,
                             BLOCK_C=256, num_warps=4)
    else:  # "96"
        _ln_kernel_96[(N,)](x, weight, bias, out, C, 1e-5,
                            BLOCK_C=128, num_warps=4)
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper – IDENTICAL across all three pass files
# ---------------------------------------------------------------------------

@torch.fx.wrap
def shared_dispatch(in_0, in_1, in_2, route):
    return _run_kernel(in_1, in_0, in_2, route)


def replacement_func():
    return shared_dispatch