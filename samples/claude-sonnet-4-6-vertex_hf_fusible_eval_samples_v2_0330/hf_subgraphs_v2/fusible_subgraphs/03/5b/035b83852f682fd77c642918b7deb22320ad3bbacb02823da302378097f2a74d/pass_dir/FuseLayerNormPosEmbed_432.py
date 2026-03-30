import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton layer-norm kernel  –  N=432, one CTA per row, float32 accumulation
# ---------------------------------------------------------------------------
@triton.jit
def _layer_norm_kernel_432(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_row,
    eps,
    N: tl.constexpr,       # = 432 (compile-time)
    BLOCK_N: tl.constexpr, # = 512 >= N
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * stride_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load row – masked positions get 0.0 (don't contribute to sum)
    x     = tl.load(X_row + cols, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean (zeros at masked positions do not affect the sum)
    mean = tl.sum(x_f32, axis=0) * (1.0 / N)

    # Variance (zero-out masked positions before squaring)
    x_c = tl.where(mask, x_f32 - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) * (1.0 / N)

    # Normalize → scale → shift → store
    inv_std = tl.rsqrt(var + eps)
    y = x_c * inv_std * w + b
    tl.store(Y_row + cols, y.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper callable returned by replacement_func()
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_layer_norm_432(in_0, in_1, in_2):
    """
    in_0 : bias   [432]  (any dtype / device)
    in_1 : weight [432]  (any dtype / device)
    in_2 : input  [1, 196, 432] on CUDA
    """
    x = in_2
    device = x.device
    # non_blocking avoids stalling the CPU while waiting for H2D transfer;
    # the CUDA stream serialises the transfer before the kernel launch.
    w = in_1.to(device=device, non_blocking=True)
    b = in_0.to(device=device, non_blocking=True)

    M = x.numel() // 432      # = 196 rows
    out = torch.empty_like(x)

    _layer_norm_kernel_432[(M,)](
        x, w, b, out,
        x.stride(-2),          # stride between rows (= 432 for contiguous)
        1e-6,                  # eps
        N=432,
        BLOCK_N=512,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface  (pattern / replacement_args / replacement_func)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    """Match layer_norm with normalized_shape=(432,)."""
    result = torch.nn.functional.layer_norm(in_2, (432,), in_1, in_0, 1e-06)
    return result


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_layer_norm_432