import torch
import triton
import triton.language as tl

def pattern(in_5, in_6, in_1, in_2):
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6

def replacement_args(in_5, in_6, in_1, in_2):
    return (in_5, in_6, in_1, in_2)

@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load inputs and compute add in float32 for numerical stability
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    xy = x + y

    # Compute mean
    mean = tl.sum(xy, axis=0) / n_cols

    # Compute variance
    diff = xy - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    out = normalized * weight + bias

    # Store output
    tl.store(out_ptr + row_start + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_layernorm(x, y, bias, weight):
    eps = 1e-12
    n_rows = x.shape[0] * x.shape[1]
    n_cols = x.shape[-1]

    # Create output with same shape and dtype as x
    out = torch.empty(x.shape[0], x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    fused_add_layernorm_kernel[grid](
        x_ptr=x, y_ptr=y, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
        n_rows=n_rows, n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=3,
    )

    return out

def replacement_func():
    return fused_add_layernorm