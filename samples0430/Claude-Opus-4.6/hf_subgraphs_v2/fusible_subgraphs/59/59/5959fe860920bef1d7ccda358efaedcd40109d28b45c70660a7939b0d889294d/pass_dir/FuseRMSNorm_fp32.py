import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-05
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.float32)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2, "rmsnorm_fp32")


@triton.jit
def _rmsnorm_bf16_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    H,
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load input row (bf16 -> fp32)
    x = tl.load(input_ptr + row * stride_in + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute variance (mean of squares)
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / H

    # Compute rsqrt(variance + eps)
    rstd = 1.0 / tl.sqrt(variance + 1e-6)

    # Normalize
    x_hat = x * rstd

    # Cast to bf16
    x_hat_bf16 = x_hat.to(tl.bfloat16)

    # Load weight (bf16) and multiply
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    out = w * x_hat_bf16

    # Store result
    tl.store(output_ptr + row * stride_out + offs, out, mask=mask)


@triton.jit
def _rmsnorm_fp32_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    H,
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load input row (bf16 -> fp32)
    x = tl.load(input_ptr + row * stride_in + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute variance (mean of squares)
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / H

    # Compute rsqrt(variance + eps)
    rstd = 1.0 / tl.sqrt(variance + 1e-5)

    # Normalize (stays fp32)
    x_hat = x * rstd

    # Load weight (bf16 -> fp32) and multiply
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = w * x_hat

    # Store result as fp32
    tl.store(output_ptr + row * stride_out + offs, out, mask=mask)


def _run_rmsnorm_bf16(in_0, in_2):
    shape = in_2.shape
    H = shape[-1]
    num_rows = in_2.numel() // H
    output = torch.empty(shape, dtype=torch.bfloat16, device=in_2.device)
    _rmsnorm_bf16_kernel[(num_rows,)](
        in_2, in_0, output, H, H, H,
        BLOCK_SIZE=2048, num_warps=8,
    )
    return output


def _run_rmsnorm_fp32(in_0, in_2):
    shape = in_2.shape
    H = shape[-1]
    num_rows = in_2.numel() // H
    output = torch.empty(shape, dtype=torch.float32, device=in_2.device)
    _rmsnorm_fp32_kernel[(num_rows,)](
        in_2, in_0, output, H, H, H,
        BLOCK_SIZE=2048, num_warps=8,
    )
    return output


@torch.fx.wrap
def rmsnorm_dispatch(in_0, in_2, route):
    if route == "rmsnorm_bf16":
        return _run_rmsnorm_bf16(in_0, in_2)
    elif route == "rmsnorm_fp32":
        return _run_rmsnorm_fp32(in_0, in_2)
    return _run_rmsnorm_bf16(in_0, in_2)


def replacement_func():
    return rmsnorm_dispatch