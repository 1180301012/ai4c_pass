import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8),
    ],
    key=["cols"],
)
@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    rows_strides,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * rows_strides
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = row_offset + col_offsets
    mask = col_offsets < cols
    vals = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
    max_val = tl.max(vals)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    tl.store(output_ptr + offsets, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_softmax_wrapper(in_0, in_1):
    batch_size, seq_len, _ = in_0.shape
    out_softmax = torch.empty_like(in_0)
    grid = (batch_size * seq_len,)
    fused_softmax_kernel[grid](in_0, out_softmax, in_0.stride(0), in_0.shape[-1])
    out_view = in_1.view(2, 512, -1)
    return (out_softmax, out_view)


def pattern(in_0, in_1):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(2, 512, -1)
    return (tmp_4, tmp_5)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_wrapper