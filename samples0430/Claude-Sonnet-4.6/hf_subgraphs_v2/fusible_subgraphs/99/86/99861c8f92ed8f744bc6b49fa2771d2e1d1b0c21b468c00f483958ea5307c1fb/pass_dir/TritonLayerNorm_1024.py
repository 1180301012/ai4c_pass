import torch
import triton
import triton.language as tl


@triton.jit
def _triton_layer_norm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    # Load input row and affine params
    x = tl.load(x_ptr + row * N + cols)
    w = tl.load(w_ptr + cols)
    b = tl.load(b_ptr + cols)

    # Upcast to float32 for numerical stability
    x_f32 = x.to(tl.float32)

    # Mean
    mean = tl.sum(x_f32, axis=0) / BLOCK_N
    diff = x_f32 - mean

    # Variance
    var  = tl.sum(diff * diff, axis=0) / BLOCK_N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Normalize + affine transform
    norm = diff * rstd
    out  = norm * w.to(tl.float32) + b.to(tl.float32)

    # Store in original dtype
    if IS_BF16:
        tl.store(out_ptr + row * N + cols, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row * N + cols, out.to(tl.float16))


@torch.fx.wrap
def triton_layer_norm(bias, weight, x):
    """Drop-in replacement for layer_norm(x, (1024,), weight, bias, 1e-05)."""
    N       = 1024
    M       = x.numel() // N
    is_bf16 = (x.dtype == torch.bfloat16)
    out     = torch.empty_like(x)
    _triton_layer_norm_kernel[(M,)](
        x, weight, bias, out,
        N,
        IS_BF16=is_bf16,
        BLOCK_N=1024,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Single-output pattern: matches layer_norm(x, (1024,), weight, bias, 1e-05)
# Works for all 4 target graphs regardless of return order of the outer model.
# ---------------------------------------------------------------------------
def pattern(bias, weight, x):
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)


def replacement_args(bias, weight, x):
    return (bias, weight, x)


def replacement_func():
    return triton_layer_norm