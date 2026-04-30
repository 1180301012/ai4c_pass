import torch
import triton
import triton.language as tl

_NEG_INF = -3.4028234663852886e+38


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_warps=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2),
    ],
    key=["seq_len"],
)
@triton.jit
def causal_mask_kernel(
    in_ptr,
    out_ptr,
    seq_len,
    stride_in_b,
    stride_in_s,
    stride_out_b,
    stride_out_h,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len
    valid = mask_m[:, None] & mask_n[None, :]

    in_vals = tl.load(
        in_ptr + offs_n[None, :] * stride_in_s,
        mask=mask_n[None, :],
        other=0,
    )

    future = offs_n[None, :] > offs_m[:, None]
    masked_by_attention = in_vals == 0
    keep_row = tl.sum(in_vals.to(tl.int32), axis=1) > 0

    neg_inf = tl.full([BLOCK_M, BLOCK_N], _NEG_INF, tl.float32)
    zeros = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)

    out = tl.where(future | masked_by_attention, neg_inf, zeros)
    out = tl.where(keep_row[:, None], out, zeros)

    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_out_m
        + offs_n[None, :] * stride_out_n
    )
    tl.store(out_ptrs, out, mask=valid)


@torch.fx.wrap
def causal_mask_wrapper(in_0):
    seq_len = in_0.shape[-1]
    out = torch.empty((1, 1, seq_len, seq_len), device=in_0.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]) * triton.cdiv(seq_len, meta["BLOCK_N"]),)

    causal_mask_kernel[grid](
        in_0,
        out,
        seq_len,
        in_0.stride(0),
        in_0.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out