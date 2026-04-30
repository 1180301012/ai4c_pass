import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["rows"],
)
@triton.jit
def layernorm_1024_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    offsets = row * 1024 + cols

    x = tl.load(x_ptr + offsets)
    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) * (1.0 / 1024.0)
    diff = x_fp32 - mean
    var = tl.sum(diff * diff, axis=0) * (1.0 / 1024.0)
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    out = diff * inv_std * weight + bias
    tl.store(out_ptr + offsets, out)


def _add_impl(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements)
    return out


def _layernorm_impl(x, weight, bias):
    out = torch.empty_like(x)
    rows = x.numel() // 1024
    layernorm_1024_kernel[(rows,)](x, weight, bias, out, rows, 1e-5)
    return out


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "add":
        x, y, _ = args
        return _add_impl(x, y)
    if route == "layernorm":
        x, weight, bias, _ = args
        return _layernorm_impl(x, weight, bias)
    raise ValueError(f"Unsupported route: {route}")