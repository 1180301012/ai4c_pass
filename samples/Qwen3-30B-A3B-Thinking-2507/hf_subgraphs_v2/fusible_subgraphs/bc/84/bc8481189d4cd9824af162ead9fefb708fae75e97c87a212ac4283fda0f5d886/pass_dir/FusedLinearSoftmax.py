import torch
import triton
import triton.language as tl


def pattern(in_2 : torch.Tensor, in_1 : torch.Tensor, in_0 : torch.Tensor):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return (tmp_4,)

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def fused_linear_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    seq_size,
    hidden_size,
    output_features,
    group_size,
    num_groups
):
    group_id = tl.program_id(0)
    seq_id = group_id // 2
    group_in_seq = group_id % 2
    feature_start = group_in_seq * group_size

    input_row = tl.load(
        input_ptr + seq_id * hidden_size,
        mask=tl.arange(0, hidden_size) < hidden_size,
        other=0.0,
        dtype=tl.float32
    )

    bias_vals = tl.load(
        bias_ptr + feature_start + tl.arange(0, group_size),
        mask=tl.arange(0, group_size) < group_size,
        other=0.0,
        dtype=tl.float32
    )

    weight_offsets = weight_ptr + (feature_start * hidden_size)
    weight_rows = tl.load(
        weight_offsets + tl.arange(0, group_size)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :],
        mask=(tl.arange(0, group_size)[:, None] < group_size) & (tl.arange(0, hidden_size)[None, :] < hidden_size),
        other=0.0,
        dtype=tl.float32
    )

    linear_vals = tl.dot(input_row, weight_rows.T) + bias_vals

    max_val = tl.max(linear_vals)
    exp_vals = tl.exp(linear_vals - max_val)
    sum_val = tl.sum(exp_vals)
    softmax_vals = exp_vals / sum_val

    tl.store(
        output_ptr + group_id * group_size,
        softmax_vals,
        mask=tl.arange(0, group_size) < group_size,
        dtype=tl.float32
    )


@torch.fx.wrap
def fused_linear_softmax(in_2, in_1, in_0):
    batch_size, seq_size, hidden_size = in_2.shape
    output_features = in_1.shape[0]
    group_size = 9

    num_groups = (batch_size * seq_size * output_features) // group_size

    output = torch.empty((num_groups, group_size, 1), dtype=in_2.dtype, device=in_2.device)

    grid = (num_groups,)
    fused_linear_softmax_kernel[grid](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        seq_size=seq_size,
        hidden_size=hidden_size,
        output_features=output_features,
        group_size=group_size,
        num_groups=num_groups
    )

    return output

def replacement_func():
    return fused_linear_softmax