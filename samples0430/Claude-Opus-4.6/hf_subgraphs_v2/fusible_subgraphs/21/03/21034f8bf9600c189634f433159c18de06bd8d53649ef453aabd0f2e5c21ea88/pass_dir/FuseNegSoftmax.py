import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
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

    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    x = tl.load(input_ptrs, mask=mask, other=float('inf')).to(tl.float32)

    # Compute softmax(-x) which is equivalent to softmax(max(x) - x)
    # since softmax is shift-invariant: softmax(y + c) = softmax(y)
    neg_x = -x
    max_val = tl.max(neg_x, axis=0)
    shifted = neg_x - max_val
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    result = exp_vals / sum_exp

    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, result, mask=mask)


@torch.fx.wrap
def fused_neg_softmax(in_0):
    shape = in_0.shape
    n_rows = 1
    for s in shape[:-1]:
        n_rows *= s
    n_cols = shape[-1]

    output = torch.empty_like(in_0)

    grid = (n_rows,)
    fused_neg_softmax_kernel[grid](
        in_0,
        output,
        n_cols,
        in_0.stride(-2),
        output.stride(-2),
    )

    return output


def replacement_func():
    return fused_neg_softmax