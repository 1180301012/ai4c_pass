import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "fused_add_layernorm")


@triton.jit
def fused_add_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load inputs and convert to float32 for accuracy
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # Add
    sum_xy = x + y

    # Compute mean
    mean = tl.sum(sum_xy, axis=0) / n_cols

    # Compute variance (two-pass for numerical stability)
    diff = sum_xy - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Scale and shift
    result = normalized * weight + bias

    # Store
    tl.store(out_ptr + row_start + offsets, result, mask=mask)


# Shared dispatch wrapper for both passes
@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "identity_reshape":
        # Just return the input - reshape/permute chain is identity
        return args[0]
    elif route == "fused_add_layernorm":
        bias, weight, x, y = args[0], args[1], args[2], args[3]
        n_rows = x.shape[0] * x.shape[1]
        n_cols = x.shape[-1]
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        fused_add_layernorm_kernel[grid](
            bias_ptr=bias,
            weight_ptr=weight,
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_rows=n_rows,
            n_cols=n_cols,
            eps=1e-05,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return out
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper