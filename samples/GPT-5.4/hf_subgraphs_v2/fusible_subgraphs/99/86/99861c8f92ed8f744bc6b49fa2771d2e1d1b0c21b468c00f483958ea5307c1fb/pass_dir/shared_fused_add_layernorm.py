import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    ln_out_ptr,
    rows,
    x_s0,
    x_s1,
    x_s2,
    y_s0,
    y_s1,
    y_s2,
    add_s0,
    add_s1,
    add_s2,
    ln_s0,
    ln_s1,
    ln_s2,
    eps,
    N_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid // rows
    row1 = pid % rows
    offs = tl.arange(0, N_COLS)

    x_ptrs = x_ptr + row0 * x_s0 + row1 * x_s1 + offs * x_s2
    y_ptrs = y_ptr + row0 * y_s0 + row1 * y_s1 + offs * y_s2
    add_ptrs = add_out_ptr + row0 * add_s0 + row1 * add_s1 + offs * add_s2
    ln_ptrs = ln_out_ptr + row0 * ln_s0 + row1 * ln_s1 + offs * ln_s2

    x = tl.load(x_ptrs).to(tl.float32)
    y = tl.load(y_ptrs).to(tl.float32)
    summed = x + y
    tl.store(add_ptrs, summed)

    mean = tl.sum(summed, axis=0) / N_COLS
    centered = summed - mean
    var = tl.sum(centered * centered, axis=0) / N_COLS
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)
    ln = centered * rstd * weight + bias
    tl.store(ln_ptrs, ln)


@torch.fx.wrap
def fused_add_layernorm_dispatch(x, y, weight, bias, route):
    add_out = torch.empty_like(x)
    ln_out = torch.empty_like(x)

    shape = x.shape
    x_stride = x.stride()
    y_stride = y.stride()
    add_stride = add_out.stride()
    ln_stride = ln_out.stride()

    rows = shape[1]
    grid = (shape[0] * rows,)

    _fused_add_layernorm_kernel[grid](
        x,
        y,
        weight,
        bias,
        add_out,
        ln_out,
        rows,
        x_stride[0],
        x_stride[1],
        x_stride[2],
        y_stride[0],
        y_stride[1],
        y_stride[2],
        add_stride[0],
        add_stride[1],
        add_stride[2],
        ln_stride[0],
        ln_stride[1],
        ln_stride[2],
        1e-05,
        N_COLS=1024,
        num_warps=8,
        num_stages=2,
    )

    if route == "tmp2_tmp4":
        return (add_out, ln_out)
    return (ln_out, add_out)