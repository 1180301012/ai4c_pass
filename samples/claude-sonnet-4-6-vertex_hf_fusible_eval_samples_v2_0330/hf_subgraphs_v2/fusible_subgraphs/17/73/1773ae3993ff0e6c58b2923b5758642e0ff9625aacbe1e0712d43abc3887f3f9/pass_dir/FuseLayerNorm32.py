import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_32_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_row = x_ptr + row_idx * N
    out_row = out_ptr + row_idx * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_row + cols, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    mean = tl.sum(x_f32, axis=0) / N

    diff = tl.where(mask, x_f32 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    out = diff * rstd * w + b

    tl.store(out_row + cols, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_layer_norm_32(x, weight, bias):
    orig_shape = x.shape
    x_2d = x.view(-1, 32)
    M = x_2d.shape[0]
    N = 32

    out = torch.empty_like(x_2d)

    _layer_norm_32_kernel[(M,)](
        x_2d, weight, bias, out,
        M, N,
        1e-12,
        BLOCK_SIZE=32,
    )

    return out.view(orig_shape)


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (32,), weight, bias, 1e-12)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return triton_layer_norm_32