import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 16)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_layernorm_kernel_16(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    out_sum_ptr, out_norm_ptr,
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

    # Load inputs and cast to float32 for numerical stability
    x = tl.load(x_ptr + offsets, mask=row_mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=row_mask, other=0.0).to(tl.float32)

    # Add
    sum_val = x + y

    # Compute mean (masked elements are 0, so sum is correct; divide by real count)
    mean = tl.sum(sum_val, axis=0) / HIDDEN_SIZE

    # Compute variance (zero out masked elements in diff^2)
    diff = sum_val - mean
    diff_sq = diff * diff * row_mask.to(tl.float32)
    var = tl.sum(diff_sq, axis=0) / HIDDEN_SIZE

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    norm_val = diff * rstd

    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=row_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=row_mask, other=0.0).to(tl.float32)

    # Scale and shift
    out_norm = norm_val * w + b

    # Store results (only valid elements)
    tl.store(out_sum_ptr + offsets, sum_val, mask=row_mask)
    tl.store(out_norm_ptr + offsets, out_norm, mask=row_mask)


@torch.fx.wrap
def fused_add_reshape_layernorm_16(in_0, in_1, in_2, in_3):
    HIDDEN_SIZE = 16
    BLOCK_SIZE = 32  # Next power of 2 >= 16
    n_rows = in_2.numel() // HIDDEN_SIZE

    out_sum = torch.empty((n_rows, HIDDEN_SIZE), dtype=in_2.dtype, device=in_2.device)
    out_norm = torch.empty((n_rows, HIDDEN_SIZE), dtype=in_2.dtype, device=in_2.device)

    grid = (n_rows,)

    fused_add_layernorm_kernel_16[grid](
        x_ptr=in_2, y_ptr=in_3, weight_ptr=in_1, bias_ptr=in_0,
        out_sum_ptr=out_sum, out_norm_ptr=out_norm,
        n_rows=n_rows,
        HIDDEN_SIZE=HIDDEN_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05,
    )

    return (out_sum, out_norm)


def replacement_func():
    return fused_add_reshape_layernorm_16