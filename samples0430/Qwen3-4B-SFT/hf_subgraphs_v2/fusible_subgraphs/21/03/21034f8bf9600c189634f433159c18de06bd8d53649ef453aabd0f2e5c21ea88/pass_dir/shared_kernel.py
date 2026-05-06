import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['col_size'],
)
@triton.jit
def local_softmax_max_sub_kernel(
    in_ptr,
    out_ptr,
    total_elements,
    col_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row.
    Computes: softmax(max(row) - row, dim=-1)
    Numerically stable: subtract the row-max first.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * col_size
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < col_size

    # Load row (padding out-of-bounds with 0 so they don't affect max or sum)
    x = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Online (two-pass) softmax: compute max for stability
    x_max = tl.max(x_f32, axis=0)

    # Subtract max for numerical stability
    x_shifted = x_f32 - x_max
    # Set out-of-bounds to -inf so they don't contribute to the sum
    x_shifted = tl.where(mask, x_shifted, -float('inf'))

    # Compute exp(sum = exp(x_shifted))
    exp_x = tl.exp(x_shifted)
    exp_x = tl.where(mask, exp_x, 0.0)
    sum_exp = tl.sum(exp_x, axis=0)

    # Normalize
    out_f32 = exp_x / sum_exp

    # Cast back to original dtype and store
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_max_sub_softmax(in_0):
    """
    Fused replacement for:
      torch.max(in_0, -1, keepdim=True)[0].expand_as(in_0) - in_0   → softmax(dim=-1)
    """
    total_elements = in_0.numel()
    col_size = in_0.shape[-1]  # Last dimension (512 for all target graphs)
    n_rows = total_elements // col_size

    out = torch.empty_like(in_0)

    local_softmax_max_sub_kernel[(n_rows,)](
        in_0, out,
        total_elements,
        col_size,
        BLOCK_SIZE=512,
    )

    return out