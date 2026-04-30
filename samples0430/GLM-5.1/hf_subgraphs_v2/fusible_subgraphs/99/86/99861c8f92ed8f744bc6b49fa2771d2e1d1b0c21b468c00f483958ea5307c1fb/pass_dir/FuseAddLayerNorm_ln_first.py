import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_ln_kernel(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    add_out_ptr, ln_out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs and convert to float32 for numerical stability
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused add
    add_result = x + y

    # Layer norm: compute mean
    mean = tl.sum(tl.where(mask, add_result, 0.0), axis=0) / n_cols
    # Layer norm: compute variance
    diff = add_result - mean
    var = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0) / n_cols
    # Layer norm: normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Scale and shift
    ln_result = weight * normalized + bias_val

    # Store results (Triton handles dtype casting back to original dtype)
    tl.store(add_out_ptr + row_start + col_offsets, add_result, mask=mask)
    tl.store(ln_out_ptr + row_start + col_offsets, ln_result, mask=mask)


@torch.fx.wrap
def fused_add_ln_dispatch(bias, weight, x, y, route):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    eps = 1e-05

    add_out = torch.empty_like(x)
    ln_out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    fused_add_ln_kernel[grid](
        x_ptr=x, y_ptr=y,
        weight_ptr=weight, bias_ptr=bias,
        add_out_ptr=add_out, ln_out_ptr=ln_out,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if route == "add_first":
        return (add_out, ln_out)
    elif route == "ln_first":
        return (ln_out, add_out)
    else:
        raise ValueError(f"Unknown route: {route}")


def pattern(bias, weight, x, y):
    add_result = x + y
    ln_result = torch.nn.functional.layer_norm(add_result, (1024,), weight, bias, 1e-05)
    return (ln_result, add_result)


def replacement_args(bias, weight, x, y):
    return (bias, weight, x, y, "ln_first")


def replacement_func():
    return fused_add_ln_dispatch