import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────
# Pattern: layer_norm with normalized_shape = (768,)
# ──────────────────────────────────────────────────────────────
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-12)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ──────────────────────────────────────────────────────────────
# Triton kernel  (single-pass: all configs have BLOCK_SIZE >= 768)
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def _layer_norm_kernel_768(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N,
    stride_x, stride_y,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Y_row = Y_ptr + row * stride_y

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load & upcast to float32 for numerical stability
    x = tl.load(X_row + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Mean  (out-of-bounds loaded as 0.0, do not affect sum / N)
    mean = tl.sum(x, axis=0) / N

    # Variance
    x_c = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y = x_c * rstd * w + b

    # Triton automatically down-casts fp32 → output pointer dtype (bf16/fp16/fp32)
    tl.store(Y_row + offsets, y, mask=mask)


# ──────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_layer_norm_768(x, weight, bias):
    orig_shape = x.shape
    N = weight.numel()            # == 768
    M = x.numel() // N

    x_flat = x.contiguous().view(M, N)
    # Output preserves input dtype; Triton stores fp32 → native dtype in kernel
    y_flat = torch.empty_like(x_flat)

    _layer_norm_kernel_768[(M,)](
        X_ptr=x_flat,
        W_ptr=weight,
        B_ptr=bias,
        Y_ptr=y_flat,
        M=M,
        N=N,
        stride_x=x_flat.stride(0),
        stride_y=y_flat.stride(0),
        eps=1e-12,
    )

    return y_flat.view(orig_shape)


# ──────────────────────────────────────────────────────────────
# Required entry-point
# ──────────────────────────────────────────────────────────────
def replacement_func():
    return triton_layer_norm_768