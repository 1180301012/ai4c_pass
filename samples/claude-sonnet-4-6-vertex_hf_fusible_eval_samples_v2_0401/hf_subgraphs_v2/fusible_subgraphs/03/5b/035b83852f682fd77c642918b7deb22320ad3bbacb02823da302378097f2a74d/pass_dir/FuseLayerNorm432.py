import torch
import triton
import triton.language as tl


@triton.jit
def _ln_432_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_orig = tl.load(X_row + cols, mask=mask, other=0.0)
    x_f32  = x_orig.to(tl.float32)

    mean   = tl.sum(x_f32, axis=0) / N
    x_c    = tl.where(mask, x_f32 - mean, 0.0)
    var    = tl.sum(x_c * x_c, axis=0) / N
    rstd   = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_c * rstd * w + b
    tl.store(Y_row + cols, y.to(x_orig.dtype), mask=mask)


@torch.fx.wrap
def triton_layer_norm_432(in_0, in_1, in_2):
    """bias, weight, input → layer_norm output"""
    M = in_2.numel() // 432
    out = torch.empty_like(in_2)
    _ln_432_kernel[(M,)](
        in_2, in_1, in_0, out,
        432, 1e-6,
        BLOCK_SIZE=512, num_warps=8,
    )
    return out


def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (432,), in_1, in_0, 1e-06)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_layer_norm_432