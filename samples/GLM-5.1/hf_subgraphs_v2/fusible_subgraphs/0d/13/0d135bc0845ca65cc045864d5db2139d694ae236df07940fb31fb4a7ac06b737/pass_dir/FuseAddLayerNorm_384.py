import torch
import triton
import triton.language as tl


def pattern(in_5, in_6, in_1, in_2):
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_5, in_6, in_1, in_2):
    return (in_5, in_6, in_2, in_1)


@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, y_ptr,
    w_ptr, b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_off = row_idx * n_cols

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row_off + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_off + offsets, mask=mask, other=0.0)
    xy = x + y

    xy_f = xy.to(tl.float32)
    mean = tl.sum(tl.where(mask, xy_f, 0.0), axis=0) / n_cols
    diff = tl.where(mask, xy_f - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = rstd * w
    bias_fused = -mean * rstd * w + b

    result = tl.where(mask, xy_f * scale + bias_fused, 0.0).to(xy.dtype)
    tl.store(out_ptr + row_off + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_layernorm(in_5, in_6, in_2, in_1):
    shape = in_5.shape
    n_rows = shape[0] * shape[1]
    n_cols = shape[2]

    out = torch.empty_like(in_5)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    eps = 1e-12

    grid = (n_rows,)
    fused_add_layernorm_kernel[grid](
        x_ptr=in_5, y_ptr=in_6,
        w_ptr=in_2, b_ptr=in_1,
        out_ptr=out,
        n_rows=n_rows, n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_add_layernorm