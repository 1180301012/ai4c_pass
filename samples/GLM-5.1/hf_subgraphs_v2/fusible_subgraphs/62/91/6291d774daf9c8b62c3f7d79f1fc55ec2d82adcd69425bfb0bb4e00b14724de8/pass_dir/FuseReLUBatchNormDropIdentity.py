import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # Reorder: input tensor first, then BN parameters
    return (in_4, in_0, in_1, in_2, in_3)


@triton.jit
def fused_relu_bn_kernel(
    input_ptr, output_ptr,
    running_mean_ptr, running_var_ptr, bias_ptr, weight_ptr,
    n_rows,
    n_features: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < n_rows

    feat_offsets = tl.arange(0, n_features)

    # Load BN parameters and upcast to float32 for numerical precision
    running_mean = tl.load(running_mean_ptr + feat_offsets).to(tl.float32)
    running_var = tl.load(running_var_ptr + feat_offsets).to(tl.float32)
    weight = tl.load(weight_ptr + feat_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + feat_offsets).to(tl.float32)

    # Compute scale and offset in float32:
    # scale = weight / sqrt(running_var + eps)
    # offset = bias - running_mean * scale
    scale = weight / tl.sqrt(running_var + eps)
    offset = bias - running_mean * scale

    # Load input block and upcast to float32
    input_ptrs = input_ptr + row_offsets[:, None] * n_features + feat_offsets[None, :]
    x = tl.load(input_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    # Fused computation: relu(scale * x + offset) = max(0, scale*x + offset)
    output = tl.maximum(0.0, x * scale[None, :] + offset[None, :])

    # Store output (auto-cast to output tensor dtype)
    output_ptrs = output_ptr + row_offsets[:, None] * n_features + feat_offsets[None, :]
    tl.store(output_ptrs, output, mask=row_mask[:, None])


@torch.fx.wrap
def fused_relu_bn_drop_identity(input, running_mean, running_var, bias, weight):
    n_rows = input.shape[0]
    n_features = input.shape[1]
    eps = 1e-05

    output = torch.empty_like(input)

    BLOCK_M = 8
    num_programs = (n_rows + BLOCK_M - 1) // BLOCK_M

    fused_relu_bn_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bias_ptr=bias,
        weight_ptr=weight,
        n_rows=n_rows,
        n_features=n_features,
        eps=eps,
        BLOCK_M=BLOCK_M,
    )

    return output


def replacement_func():
    return fused_relu_bn_drop_identity