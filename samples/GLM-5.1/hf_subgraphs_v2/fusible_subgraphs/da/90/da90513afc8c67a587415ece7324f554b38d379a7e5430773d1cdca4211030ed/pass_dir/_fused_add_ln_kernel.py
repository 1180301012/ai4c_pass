import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_ln_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Out_ptr,
    N_rows, eps,
    N_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * N_cols

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_cols
    x_vals = tl.load(X_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    y_vals = tl.load(Y_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    vals = x_vals + y_vals

    # Masked reduction: only count valid elements
    _sum = tl.where(mask, vals, 0.0)
    _sum_sq = tl.where(mask, vals * vals, 0.0)
    mean = tl.sum(_sum, axis=0) / N_cols
    var = tl.sum(_sum_sq, axis=0) / N_cols - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(mask, w * (vals - mean) * rstd + b, 0.0)
    tl.store(Out_ptr + row_offset + offsets, out, mask=mask)


@triton.jit
def fused_add_ln_kernel_grouped(
    X_ptr, Y_ptr, W_ptr, B_ptr, Out_ptr,
    N_rows, eps,
    N_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    row_start = group_id * GROUP_SIZE
    row_end = tl.minimum(row_start + GROUP_SIZE, N_rows)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_cols

    w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    for row in range(row_start, row_end):
        row_offset = row * N_cols
        x_vals = tl.load(X_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        y_vals = tl.load(Y_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        vals = x_vals + y_vals

        _sum = tl.where(mask, vals, 0.0)
        _sum_sq = tl.where(mask, vals * vals, 0.0)
        mean = tl.sum(_sum, axis=0) / N_cols
        var = tl.sum(_sum_sq, axis=0) / N_cols - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        out = tl.where(mask, w * (vals - mean) * rstd + b, 0.0)
        tl.store(Out_ptr + row_offset + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm_dispatch(bias, weight, x, y, route=""):
    """Shared dispatch wrapper for all fused add+layer_norm passes.
    Uses specialized kernel configurations for each normalized shape size.
    BLOCK_SIZE is next power of 2 >= N_cols for Triton compatibility.
    """
    N_cols = weight.shape[0]
    total_elements = x.numel()
    N_rows = total_elements // N_cols

    out = torch.empty_like(x)

    if route == "route_768":
        grid = (N_rows,)
        fused_add_ln_kernel[grid](
            x, y, weight, bias, out,
            N_rows, 1e-05,
            N_cols=768, BLOCK_SIZE=1024,
        )
    elif route == "route_1024":
        grid = (N_rows,)
        fused_add_ln_kernel[grid](
            x, y, weight, bias, out,
            N_rows, 1e-05,
            N_cols=1024, BLOCK_SIZE=1024,
        )
    elif route == "route_16":
        GROUP_SIZE = 128
        grid = ((N_rows + GROUP_SIZE - 1) // GROUP_SIZE,)
        fused_add_ln_kernel_grouped[grid](
            x, y, weight, bias, out,
            N_rows, 1e-05,
            N_cols=16, BLOCK_SIZE=16, GROUP_SIZE=GROUP_SIZE,
        )

    return out