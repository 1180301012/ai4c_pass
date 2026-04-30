import torch
import triton
import triton.language as tl


# ========== Triton Layer Norm Kernel ==========
@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    offset = row * N + cols
    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std

    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    tl.store(Y_ptr + offset, y, mask=mask)


@torch.fx.wrap
def optimized_layer_norm(in_0, in_1, in_2):
    N = in_2.shape[-1]
    n_rows = in_2.shape[0] * in_2.shape[1]
    y = torch.empty_like(in_2)
    BLOCK_SIZE = 512 if N > 256 else 256
    _layer_norm_fwd_kernel[(n_rows,)](
        in_2, y, in_1, in_0,
        N, 1e-06,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ========== Pattern: layer_norm with normalized_shape as wildcard input ==========
def pattern(in_0, in_1, in_2, ns):
    tmp_2 = torch.nn.functional.layer_norm(in_2, ns, in_1, in_0, 1e-06)
    return tmp_2


def replacement_args(in_0, in_1, in_2, ns):
    return (in_0, in_1, in_2)


def replacement_func():
    return optimized_layer_norm