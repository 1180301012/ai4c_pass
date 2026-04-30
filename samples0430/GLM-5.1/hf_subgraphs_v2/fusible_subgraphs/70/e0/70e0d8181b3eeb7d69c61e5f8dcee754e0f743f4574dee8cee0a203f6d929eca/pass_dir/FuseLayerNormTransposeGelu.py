import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_layernorm_transpose_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N_rows, N_cols, eps,
    input_stride_1, input_stride_2,
    output_stride_1, output_stride_2,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles ROWS_PER_PROGRAM rows of the input
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM
    row_offsets = row_start + tl.arange(0, ROWS_PER_PROGRAM)
    row_mask = row_offsets < N_rows

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < N_cols

    # 2D mask for loads/stores
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input block [ROWS_PER_PROGRAM, BLOCK_SIZE]
    input_ptrs = input_ptr + row_offsets[:, None] * input_stride_1 + col_offsets[None, :] * input_stride_2
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm: compute mean per row
    mean = tl.sum(x, axis=1) / N_cols  # [ROWS_PER_PROGRAM]
    x_centered = x - mean[:, None]

    # LayerNorm: compute variance per row
    var = tl.sum(x_centered * x_centered, axis=1) / N_cols  # [ROWS_PER_PROGRAM]
    rstd = 1.0 / tl.sqrt(var + eps)  # [ROWS_PER_PROGRAM]

    # Load weight [N_cols] and bias [N_cols]
    weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Apply affine transform: (x - mean) * rstd * weight + bias
    normalized = x_centered * rstd[:, None] * weight[None, :] + bias[None, :]

    # GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    gelu_out = 0.5 * normalized * (1.0 + tl.math.erf(normalized / sqrt2))

    # Store in transposed position: output[j, i] where j=col, i=row
    # Output shape is [1, N_cols, N_rows], so:
    # output[0, col_offsets[j], row_offsets[i]] = output_ptr + col_offsets[j] * output_stride_1 + row_offsets[i] * output_stride_2
    output_ptrs = output_ptr + col_offsets[None, :] * output_stride_1 + row_offsets[:, None] * output_stride_2
    tl.store(output_ptrs, gelu_out, mask=mask)


@torch.fx.wrap
def fused_layernorm_transpose_gelu(bias, weight, input):
    # Input shape: [B, N_rows, N_cols] = [1, 3999, 512]
    # Output shape: [B, N_cols, N_rows] = [1, 512, 3999]
    N_rows = input.shape[1]
    N_cols = input.shape[2]
    eps = 1e-05

    B = input.shape[0]
    output = torch.empty((B, N_cols, N_rows), dtype=input.dtype, device=input.device)

    ROWS_PER_PROGRAM = 4
    BLOCK_SIZE = triton.next_power_of_2(N_cols)

    num_programs = (N_rows + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM

    fused_layernorm_transpose_gelu_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N_rows=N_rows, N_cols=N_cols, eps=eps,
        input_stride_1=input.stride(1), input_stride_2=input.stride(2),
        output_stride_1=output.stride(1), output_stride_2=output.stride(2),
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_layernorm_transpose_gelu