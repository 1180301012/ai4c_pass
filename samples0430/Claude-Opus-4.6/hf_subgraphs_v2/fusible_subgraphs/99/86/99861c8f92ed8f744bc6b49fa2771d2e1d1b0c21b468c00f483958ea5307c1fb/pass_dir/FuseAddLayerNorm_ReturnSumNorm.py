import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)


def replacement_args(bias, weight, x):
    return (x, weight, bias)


@triton.jit
def _layernorm_fwd_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_x_row,
    N_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, N_COLS)
    off = row * stride_x_row + cols

    # Load row and promote to float32
    x = tl.load(X_ptr + off).to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) * (1.0 / N_COLS)

    # Variance
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) * (1.0 / N_COLS)

    # Normalize
    y = xc * tl.rsqrt(var + 1e-05)

    # Scale and shift
    w = tl.load(W_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)
    y = y * w + b

    # Store (Triton auto-casts to output pointer dtype)
    tl.store(Y_ptr + off, y)


@torch.fx.wrap
def triton_layernorm(x, weight, bias):
    y = torch.empty_like(x)
    n_rows = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
    stride = x.stride(-2)
    _layernorm_fwd_kernel[(n_rows,)](
        x, weight, bias, y,
        stride, x.shape[-1],
        num_warps=8, num_stages=1,
    )
    return y


def replacement_func():
    return triton_layernorm