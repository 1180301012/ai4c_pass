import torch
import triton
import triton.language as tl


def pattern(in_7, in_0, in_1, in_3, in_2):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)


@triton.jit
def batch_norm_eval_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < C

    # Load parameters and compute scale/offset in float32 for accuracy
    mean = tl.load(mean_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute scale and offset: output = input * scale + offset
    # where scale = gamma / sqrt(var + eps), offset = beta - mean * scale
    scale = gamma / tl.sqrt(var + eps)
    offset = beta - mean * scale

    # Load input row and compute in float32
    input_row = tl.load(input_ptr + row_idx * C + col_offsets, mask=mask, other=0.0).to(tl.float32)
    output = input_row * scale + offset

    # Store output
    tl.store(output_ptr + row_idx * C + col_offsets, output, mask=mask)


@torch.fx.wrap
def batch_norm_eval_triton(input, running_mean, running_var, weight, bias):
    shape = input.shape
    N = shape[0]
    C = shape[-1]
    eps = 1e-05

    output = torch.empty_like(input)

    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (N,)

    batch_norm_eval_kernel[grid](
        input_ptr=input,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C=C, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return batch_norm_eval_triton