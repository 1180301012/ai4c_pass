import torch
import triton
import triton.language as tl


@triton.jit
def _ln_relu_kernel(
    input_ptr,
    ln_w_ptr,
    ln_b_ptr,
    output_ptr,
    C,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused LayerNorm + ReLU kernel.
    Each program handles one row (one spatial position).
    Row index = program_id(0); column range = [0, BLOCK_C) with masking for C < BLOCK_C.
    """
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Load input row (native dtype) and upcast to float32 for numerically stable LN
    x = tl.load(input_ptr + row_idx * C + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm: compute mean
    x_centered = tl.where(mask, x - x.mean(), 0.0)

    # LayerNorm: compute variance
    var = tl.sum(x_centered * x_centered, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_hat = x_centered * rstd

    # Load LN affine parameters (native dtype) and upcast
    ln_w = tl.load(ln_w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform: y = x_hat * gamma + beta
    y = x_hat * ln_w + ln_b

    # Fused ReLU: z = max(0, y)
    z = tl.where(y > 0.0, y, 0.0)

    # Cast back to input dtype before storing
    if IS_FP16:
        result = z.to(tl.float16)
    elif IS_BF16:
        result = z.to(tl.bfloat16)
    else:
        result = z.to(tl.float32)

    tl.store(output_ptr + row_idx * C + cols, result, mask=mask)


def _run_ln_relu(c_in, w_ln, b_ln):
    """
    Run the fused LayerNorm + ReLU kernel.
    c_in : input tensor of shape (N, C, 1, 1)
    w_ln : layer-norm weight  (C,)
    b_ln : layer-norm bias    (C,)
    Returns normalized + activated output.
    """
    n_cols = c_in.shape[-1]
    n_rows = c_in.numel() // n_cols
    out = torch.empty_like(c_in)
    is_fp16 = (c_in.dtype == torch.float16)
    is_bf16 = (c_in.dtype == torch.bfloat16)

    # BLOCK_C must be a power-of-two and >= n_cols; also a tl.constexpr
    if n_cols <= 32:
        block_c = 32
    elif n_cols <= 64:
        block_c = 64
    elif n_cols <= 128:
        block_c = 128
    elif n_cols <= 256:
        block_c = 256
    else:
        block_c = 512

    _ln_relu_kernel[(n_rows,)](
        c_in, w_ln, b_ln, out,
        n_cols,
        1e-05,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        BLOCK_C=block_c,
    )
    return out


@torch.fx.wrap
def fused_kernel(c_in, w, b, ln_w, ln_b, route):
    """
    Shared replacement kernel for conv+layernorm+relu.
    Routes to the correct implementation based on "route" string.
    """
    if route == "C16":
        return _run_ln_relu(c_in, ln_w, ln_b)
    elif route == "C19":
        return _run_ln_relu(c_in, ln_w, ln_b)
    elif route == "C38":
        return _run_ln_relu(c_in, ln_w, ln_b)
    elif route == "C128":
        return _run_ln_relu(c_in, ln_w, ln_b)
    else:
        return _run_ln_relu(c_in, ln_w, ln_b)