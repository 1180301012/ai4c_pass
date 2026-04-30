import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    channels,
    hw,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = (offs // hw) % channels
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0).to(tl.float32)
    y = (x - mean) * tl.math.rsqrt(var + eps) * weight + bias
    tl.store(out_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _cat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    a_c,
    total_c,
    h,
    w,
    b_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    hw = h * w
    chw = total_c * hw
    a_chw = a_c * hw
    b_chw = b_c * hw

    n = offs // chw
    rem = offs - n * chw
    c = rem // hw
    rem2 = rem - c * hw
    hh = rem2 // w
    ww = rem2 - hh * w

    a_mask = mask & (c < a_c)
    b_mask = mask & (c >= a_c)

    a_offs = n * a_chw + c * hw + hh * w + ww
    x_a = tl.load(a_ptr + a_offs, mask=a_mask, other=0)

    bc = tl.where(c >= a_c, c - a_c, 0)
    b_offs = n * b_chw + bc * hw + hh * w + ww
    x_b = tl.load(b_ptr + b_offs, mask=b_mask, other=0)

    x = tl.where(c < a_c, x_a, x_b)
    tl.store(out_ptr + offs, x, mask=mask)


def _identity(x):
    return x


def _relu(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _relu_kernel[grid](x, out, n_elements)
    return out


def _bn(x, running_mean, running_var, weight, bias):
    out = torch.empty_like(x)
    n_elements = x.numel()
    channels = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _bn_kernel[grid](x, running_mean, running_var, weight, bias, out, n_elements, channels, hw, 0.001)
    return out


def _cat(a, b):
    n = a.shape[0]
    a_c = a.shape[1]
    h = a.shape[2]
    w = a.shape[3]
    b_c = b.shape[1]
    out = torch.empty((n, a_c + b_c, h, w), device=a.device, dtype=a.dtype)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _cat_kernel[grid](a, b, out, n_elements, a_c, a_c + b_c, h, w, b_c)
    return out


def shared_dispatch(*args):
    route = args[-1]
    if route == 'identity':
        return _identity(args[0])
    if route == 'relu':
        return _relu(args[0])
    if route == 'bn':
        return _bn(args[0], args[1], args[2], args[3], args[4])
    if route == 'cat':
        return _cat(args[0], args[1])
    return args[0]