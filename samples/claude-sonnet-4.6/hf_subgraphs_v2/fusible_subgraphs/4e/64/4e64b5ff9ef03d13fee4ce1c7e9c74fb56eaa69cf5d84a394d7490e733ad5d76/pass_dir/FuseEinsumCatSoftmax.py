import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: match aten-level cat + aten-level softmax (no einsum in pattern).
# The einsum node stays in the graph and feeds 'y' to our replacement.
# -----------------------------------------------------------------------
def pattern(x, y):
    tmp_2 = torch.ops.aten.cat.default([x, y], -1)
    tmp_3 = torch.ops.aten._softmax.default(tmp_2, -1, False)
    return tmp_3


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_J': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_J': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_J': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_J': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_J': 64}, num_warps=16, num_stages=2),
    ],
    key=['N_ROWS', 'J'],
)
@triton.jit
def fused_cat_softmax_kernel(
    x_ptr, y_ptr, out_ptr,
    N_ROWS, J,
    stride_x_row, stride_x_j,
    stride_y_row, stride_y_j,
    stride_out_row, stride_out_j,
    BLOCK_J: tl.constexpr,
):
    """
    Each program handles one row of 2*J elements:
      1. Load x[row, :J]  (J elements, from first input)
      2. Load y[row, :J]  (J elements, from second input / einsum result)
      3. Compute numerically-stable softmax over the 2J concatenated values
      4. Write 2J softmax values to output
    """
    row = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_J)   # [J]

    # Load x row
    x_ptrs = x_ptr + row * stride_x_row + j_offsets * stride_x_j
    x_vals = tl.load(x_ptrs).to(tl.float32)

    # Load y row
    y_ptrs = y_ptr + row * stride_y_row + j_offsets * stride_y_j
    y_vals = tl.load(y_ptrs).to(tl.float32)

    # Numerically stable softmax over [x_vals, y_vals]
    max_x = tl.max(x_vals, axis=0)
    max_y = tl.max(y_vals, axis=0)
    max_val = tl.maximum(max_x, max_y)

    exp_x = tl.exp(x_vals - max_val)
    exp_y = tl.exp(y_vals - max_val)

    sum_exp = tl.sum(exp_x, axis=0) + tl.sum(exp_y, axis=0)

    out_x = exp_x / sum_exp
    out_y = exp_y / sum_exp

    # Write output: first J from x part, next J from y part
    out_base = row * stride_out_row
    tl.store(out_ptr + out_base + j_offsets * stride_out_j, out_x)
    tl.store(out_ptr + out_base + (J + j_offsets) * stride_out_j, out_y)


@torch.fx.wrap
def fused_cat_softmax(x, y):
    # x, y: both [*, J] tensors — arbitrary leading dims, last dim = J
    # Output: [*, 2*J] — softmax over concatenated rows
    *leading, J = x.shape
    N_ROWS = 1
    for d in leading:
        N_ROWS *= d

    out = torch.empty((*leading, 2 * J), dtype=x.dtype, device=x.device)

    # Flatten to [N_ROWS, J] for the kernel
    x_flat = x.reshape(N_ROWS, J)
    y_flat = y.reshape(N_ROWS, J)
    out_flat = out.reshape(N_ROWS, 2 * J)

    fused_cat_softmax_kernel[(N_ROWS,)](
        x_flat, y_flat, out_flat,
        N_ROWS, J,
        x_flat.stride(0), x_flat.stride(1),
        y_flat.stride(0), y_flat.stride(1),
        out_flat.stride(0), out_flat.stride(1),
    )

    return out


def replacement_func():
    return fused_cat_softmax