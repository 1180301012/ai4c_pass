import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * HIDDEN_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = col_offsets < HIDDEN_SIZE
    offsets = row_start + col_offsets

    # Load input and cast to float32 for numerical stability
    x = tl.load(input_ptr + offsets, mask=row_mask, other=0.0).to(tl.float32)

    # Compute mean (masked elements are 0, so sum is correct; divide by real count)
    mean = tl.sum(x, axis=0) / HIDDEN_SIZE

    # Compute variance (zero out masked elements in diff^2 to avoid contribution)
    diff = x - mean
    diff_sq = diff * diff * row_mask.to(tl.float32)
    var = tl.sum(diff_sq, axis=0) / HIDDEN_SIZE

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    norm_val = diff * rstd

    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=row_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=row_mask, other=0.0).to(tl.float32)

    # Scale and shift
    out = norm_val * w + b

    # Store result (only valid elements)
    tl.store(output_ptr + offsets, out, mask=row_mask)


@torch.fx.wrap
def kernel_layernorm_768(input, weight, bias):
    HIDDEN_SIZE = 768
    BLOCK_SIZE = 1024  # Next power of 2 >= 768
    n_rows = input.shape[0]

    output = torch.empty_like(input)

    grid = (n_rows,)

    layernorm_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        n_rows=n_rows,
        HIDDEN_SIZE=HIDDEN_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05,
    )

    return output


@torch.fx.wrap
def kernel_layernorm_16(input, weight, bias):
    HIDDEN_SIZE = 16
    BLOCK_SIZE = 32  # Power of 2 >= 16, using 32 for warp efficiency
    n_rows = input.shape[0]

    output = torch.empty_like(input)

    grid = (n_rows,)

    layernorm_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        n_rows=n_rows,
        HIDDEN_SIZE=HIDDEN_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05,
    )

    return output


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    tensor_args = args[:-1]
    if route == "layernorm_768":
        return kernel_layernorm_768(*tensor_args)
    elif route == "layernorm_16":
        return kernel_layernorm_16(*tensor_args)
    else:
        raise ValueError(f"Unknown route: {route}")