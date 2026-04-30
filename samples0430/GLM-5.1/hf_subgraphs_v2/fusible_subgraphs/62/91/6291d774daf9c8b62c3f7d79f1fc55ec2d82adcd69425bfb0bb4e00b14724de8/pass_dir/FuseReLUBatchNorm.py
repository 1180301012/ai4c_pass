import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_relu_bn_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    num_rows, num_cols,
    eps,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(num_cols, BLOCK_COL)
    pid_row = pid // num_col_blocks
    pid_col = pid % num_col_blocks

    row_start = pid_row * BLOCK_ROW
    col_start = pid_col * BLOCK_COL

    row_offsets = row_start + tl.arange(0, BLOCK_ROW)
    col_offsets = col_start + tl.arange(0, BLOCK_COL)

    row_mask = row_offsets < num_rows
    col_mask = col_offsets < num_cols

    # Load BN parameters (cast to float32 for precision)
    mean = tl.load(running_mean_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Compute BN scale and shift
    # output = weight * (relu(x) - running_mean) / sqrt(running_var + eps) + bias
    # = (weight / sqrt(running_var + eps)) * relu(x) + (bias - running_mean * weight / sqrt(running_var + eps))
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = w * inv_std
    shift = b - mean * scale

    # Load input (cast to float32 for computation)
    offsets = row_offsets[:, None] * num_cols + col_offsets[None, :]
    mask2d = row_mask[:, None] & col_mask[None, :]

    x = tl.load(input_ptr + offsets, mask=mask2d, other=0.0).to(tl.float32)

    # Fused: ReLU + BatchNorm (dropout is identity since p=0, training=False)
    relu_x = tl.maximum(x, 0.0)
    out = relu_x * scale[None, :] + shift[None, :]

    tl.store(output_ptr + offsets, out, mask=mask2d)


@torch.fx.wrap
def fused_relu_bn(running_mean, running_var, bias, weight, input):
    # Move BN parameters to the same device as input
    device = input.device
    running_mean = running_mean.to(device)
    running_var = running_var.to(device)
    bias = bias.to(device)
    weight = weight.to(device)

    num_rows, num_cols = input.shape
    eps = 1e-05

    BLOCK_ROW = 8
    BLOCK_COL = 128

    output = torch.empty_like(input)

    num_row_blocks = triton.cdiv(num_rows, BLOCK_ROW)
    num_col_blocks = triton.cdiv(num_cols, BLOCK_COL)
    grid = (num_row_blocks * num_col_blocks,)

    fused_relu_bn_kernel[grid](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        num_rows=num_rows,
        num_cols=num_cols,
        eps=eps,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
    )

    return (output,)


def replacement_func():
    return fused_relu_bn