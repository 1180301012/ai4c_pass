import torch
import triton
import triton.language as tl


def pattern(in_0):
    max_1 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = max_1[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_neg_softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row in native dtype
    x = tl.load(
        input_ptr + row_idx * input_row_stride + col_offsets,
        mask=mask,
        other=float('-inf'),
    )

    # Upcast to float32 for numerical stability
    x_f32 = x.to(tl.float32)

    # Find max of the row (masked positions are -inf, won't affect max)
    x_max = tl.max(x_f32, axis=0)

    # Compute exp(max - x): equivalent to softmax(-x) but numerically stable
    exp_vals = tl.exp(x_max - x_f32)

    # Zero out padded positions (when BLOCK_SIZE > n_cols)
    exp_vals = tl.where(mask, exp_vals, 0.0)

    # Compute sum of exponentials
    exp_sum = tl.sum(exp_vals, axis=0)

    # Normalize
    result_f32 = exp_vals / exp_sum

    # Cast back to original dtype
    result = result_f32.to(x.dtype)

    # Store result
    tl.store(
        output_ptr + row_idx * output_row_stride + col_offsets,
        result,
        mask=mask,
    )


@torch.fx.wrap
def fused_neg_softmax(in_0):
    # in_0 shape: [B, N, M] — softmax is over the last dim M
    n_rows = in_0.numel() // in_0.shape[-1]
    n_cols = in_0.shape[-1]

    output = torch.empty_like(in_0)

    input_row_stride = in_0.stride(-2)
    output_row_stride = output.stride(-2)

    fused_neg_softmax_kernel[(n_rows,)](
        in_0,
        output,
        n_cols,
        input_row_stride,
        output_row_stride,
    )

    return output


def replacement_func():
    return fused_neg_softmax