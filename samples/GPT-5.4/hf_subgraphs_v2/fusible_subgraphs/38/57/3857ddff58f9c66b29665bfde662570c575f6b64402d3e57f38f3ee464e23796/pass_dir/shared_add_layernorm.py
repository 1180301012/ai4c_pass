import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols

    x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / n_cols
    centered = x_fp32 - mean
    var = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = centered * inv_std
    out = out * weight + bias
    tl.store(out_ptr + row_start + offs, out, mask=mask)


def _add_reshape(x, y, n_cols):
    x = unwrap_tensor(x)
    y = unwrap_tensor(y)
    n_elements = x.numel()
    out = torch.empty_like(x)
    block_size = 1024
    grid = ((n_elements + block_size - 1) // block_size,)
    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2,
    )
    return out.reshape(-1, n_cols)


def _layer_norm(x, weight, bias, n_cols):
    x = unwrap_tensor(x)
    weight = unwrap_tensor(weight)
    bias = unwrap_tensor(bias)
    m_rows = x.numel() // n_cols
    out = torch.empty_like(x)

    if n_cols <= 16:
        block_n = 16
        num_warps = 1
    elif n_cols <= 32:
        block_n = 32
        num_warps = 1
    elif n_cols <= 64:
        block_n = 64
        num_warps = 2
    elif n_cols <= 128:
        block_n = 128
        num_warps = 4
    elif n_cols <= 256:
        block_n = 256
        num_warps = 4
    elif n_cols <= 512:
        block_n = 512
        num_warps = 8
    else:
        block_n = 1024
        num_warps = 8

    _layer_norm_kernel[(m_rows,)](
        x,
        weight,
        bias,
        out,
        n_cols,
        1e-5,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "add_reshape_768":
        return _add_reshape(args[0], args[1], 768)
    if route == "layer_norm_768":
        return _layer_norm(args[0], args[1], args[2], 768)
    if route == "add_reshape_16":
        return _add_reshape(args[0], args[1], 16)
    if route == "layer_norm_16":
        return _layer_norm(args[0], args[1], args[2], 16)
    raise RuntimeError("unknown route")