import torch
import triton
import triton.language as tl


def pattern(in_3):
    mean_out = in_3.mean(-2)
    return (mean_out,)


def replacement_args(in_3):
    return (in_3,)


@triton.jit
def mean_kernel(
    in_ptr, out_ptr,
    B, R, D,
    stride_b, stride_r, stride_d,
    stride_ob, stride_od,
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_b = offs_b < B
    mask_d = offs_d < D
    acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for r_off in range(0, R):
        ptrs = in_ptr + offs_b[:, None] * stride_b + r_off * stride_r + offs_d[None, :] * stride_d
        mask = mask_b[:, None] & mask_d[None, :]
        val = tl.load(ptrs, mask=mask, other=0.0)
        acc += val
    mean_val = acc / R
    out_ptrs = out_ptr + offs_b[:, None] * stride_ob + offs_d[None, :] * stride_od
    out_mask = mask_b[:, None] & mask_d[None, :]
    tl.store(out_ptrs, mean_val, mask=out_mask)


@torch.fx.wrap
def triton_mean_dim_neg2(input):
    B = input.shape[0]
    R = input.shape[1]
    D = input.shape[2]
    output = torch.empty((B, D), device=input.device, dtype=input.dtype)
    BLOCK_B = 1
    BLOCK_D = 256
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(D, BLOCK_D))
    mean_kernel[grid](
        in_ptr=input, out_ptr=output,
        B=B, R=R, D=D,
        stride_b=input.stride(0), stride_r=input.stride(1), stride_d=input.stride(2),
        stride_ob=output.stride(0), stride_od=output.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_D=BLOCK_D,
    )
    return output


def replacement_func():
    return triton_mean_dim_neg2