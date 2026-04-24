import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_1):
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13


def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input (bfloat16) and weight (bfloat16)
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    # Compute RMS normalization
    x_fp32 = x.to(tl.float32)
    sq = x_fp32 * x_fp32
    sum_sq = tl.sum(sq, axis=0)
    rms = tl.sqrt(sum_sq / N)
    inv_rms = 1.0 / rms

    # Scale by weight + 1, then cast back to bfloat16
    w_fp32 = w.to(tl.float32)
    result = x_fp32 * inv_rms * (1.0 + w_fp32)

    tl.store(out_ptr + row * N + offsets, result.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def fused_rms_norm(tmp_2, in_1):
    N = tmp_2.shape[-1]
    num_rows = tmp_2.numel() // N
    out = torch.empty_like(tmp_2)
    _rms_norm_kernel[(num_rows,)](tmp_2, in_1, out, N, BLOCK_SIZE=2048, num_warps=4)
    return out


def replacement_func():
    return fused_rms_norm