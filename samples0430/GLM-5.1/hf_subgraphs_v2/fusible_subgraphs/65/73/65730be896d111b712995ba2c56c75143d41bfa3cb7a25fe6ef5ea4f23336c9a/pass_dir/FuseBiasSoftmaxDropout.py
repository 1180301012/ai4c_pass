import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    add_result = in_1 + in_0
    float_result = add_result.float()
    softmax_result = torch.nn.functional.softmax(float_result, dim=-1)
    type_as_result = softmax_result.type_as(add_result)
    dropout_result = torch.nn.functional.dropout(type_as_result, p=0.1, training=False)
    return dropout_result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_bias_softmax_kernel(
    bias_ptr, scores_ptr, out_ptr,
    n_cols,
    row_stride_bias,
    row_stride_scores,
    row_stride_out,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    bias_row_ptr = bias_ptr + row_idx * row_stride_bias
    scores_row_ptr = scores_ptr + row_idx * row_stride_scores
    out_row_ptr = out_ptr + row_idx * row_stride_out

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input rows and cast to INPUT_DTYPE
    bias_row = tl.load(bias_row_ptr + col_offsets, mask=mask, other=0.0).to(INPUT_DTYPE)
    scores_row = tl.load(scores_row_ptr + col_offsets, mask=mask, other=0.0).to(INPUT_DTYPE)

    # Add bias to scores
    x = bias_row + scores_row

    # Cast to float32 for softmax (numerical stability)
    x_float = x.to(tl.float32)

    # Safe softmax: subtract max, exp, sum, divide
    x_max = tl.max(x_float, axis=0)
    x_shifted = x_float - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    result_float = x_exp / x_sum

    # Cast back to original dtype
    result = result_float.to(INPUT_DTYPE)

    tl.store(out_row_ptr + col_offsets, result, mask=mask)


@torch.fx.wrap
def fused_bias_softmax(bias, scores):
    output = torch.empty_like(scores)

    shape = bias.shape
    total_rows = shape[0] * shape[1] * shape[2]
    n_cols = shape[3]

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Compute row strides for each tensor
    # For a 4D tensor [B, H, S, D], row stride = stride[2] (stride between consecutive rows in dim 2)
    # For contiguous tensors, stride[2] = D = n_cols
    row_stride_bias = bias.stride(2)
    row_stride_scores = scores.stride(2)
    row_stride_out = output.stride(2)

    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    INPUT_DTYPE = dtype_map[bias.dtype]

    fused_bias_softmax_kernel[(total_rows,)](
        bias_ptr=bias,
        scores_ptr=scores,
        out_ptr=output,
        n_cols=n_cols,
        row_stride_bias=row_stride_bias,
        row_stride_scores=row_stride_scores,
        row_stride_out=row_stride_out,
        BLOCK_SIZE=BLOCK_SIZE,
        INPUT_DTYPE=INPUT_DTYPE,
    )

    return output


def replacement_func():
    return fused_bias_softmax