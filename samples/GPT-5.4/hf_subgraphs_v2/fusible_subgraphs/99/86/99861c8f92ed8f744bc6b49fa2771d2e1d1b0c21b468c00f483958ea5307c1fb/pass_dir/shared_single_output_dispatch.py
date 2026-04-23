import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows,
    x_s0,
    x_s1,
    x_s2,
    out_s0,
    out_s1,
    out_s2,
    eps,
    N_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid // rows
    row1 = pid % rows
    offs = tl.arange(0, N_COLS)

    x_ptrs = x_ptr + row0 * x_s0 + row1 * x_s1 + offs * x_s2
    out_ptrs = out_ptr + row0 * out_s0 + row1 * out_s1 + offs * out_s2

    x = tl.load(x_ptrs).to(tl.float32)
    mean = tl.sum(x, axis=0) / N_COLS
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / N_COLS
    rstd = tl.rsqrt(var + eps)
    weight = tl.load(weight_ptr + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)
    out = centered * rstd * weight + bias
    tl.store(out_ptrs, out)


@torch.fx.wrap
def single_output_dispatch(*args):
    route = args[-1]

    if route == 0:
        x = args[0]
        y = args[1]
        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024, num_warps=4, num_stages=2)
        return out

    x = args[0]
    weight = args[1]
    bias = args[2]
    out = torch.empty_like(x)
    x_stride = x.stride()
    out_stride = out.stride()
    rows = x.shape[1]
    grid = (x.shape[0] * rows,)
    _layer_norm_kernel[grid](
        x,
        weight,
        bias,
        out,
        rows,
        x_stride[0],
        x_stride[1],
        x_stride[2],
        out_stride[0],
        out_stride[1],
        out_stride[2],
        1e-05,
        N_COLS=1024,
        num_warps=8,
        num_stages=2,
    )
    return out