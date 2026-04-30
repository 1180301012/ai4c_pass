import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@triton.jit
def _rmsnorm_bf16_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    row_offset = row * BLOCK_SIZE

    # Load input row (bf16 -> fp32) - no mask since BLOCK_SIZE == H
    x = tl.load(input_ptr + row_offset + offs).to(tl.float32)

    # Compute variance (mean of squares)
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) * (1.0 / BLOCK_SIZE)

    # Compute rsqrt(variance + eps) using hardware instruction
    rstd = tl.math.rsqrt(variance + 1e-6)

    # Normalize and cast to bf16
    x_hat = x * rstd
    x_hat_bf16 = x_hat.to(tl.bfloat16)

    # Load weight (bf16) and multiply
    w = tl.load(weight_ptr + offs)
    out = w * x_hat_bf16

    # Store result
    tl.store(output_ptr + row_offset + offs, out)


@torch.fx.wrap
def rmsnorm_bf16_fn(in_0, in_2):
    shape = in_2.shape
    H = shape[-1]
    num_rows = in_2.numel() // H
    output = torch.empty_like(in_2)

    _rmsnorm_bf16_kernel[(num_rows,)](
        in_2, in_0, output, num_rows,
        BLOCK_SIZE=H, num_warps=4, num_stages=1,
    )

    return output


def replacement_func():
    return rmsnorm_bf16_fn