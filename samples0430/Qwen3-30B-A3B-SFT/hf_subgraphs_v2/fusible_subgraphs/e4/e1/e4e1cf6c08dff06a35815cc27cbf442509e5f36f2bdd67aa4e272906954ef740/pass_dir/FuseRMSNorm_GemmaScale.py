import torch
import triton
import triton.language as tl


# Pattern takes (tmp_2, in_1) as inputs — tmp_2 is already the in_0*in_2 result.
# Single output avoids the 2-returning-nodes vs 1-copied-node assertion mismatch.
# tmp_2 stays as a placeholder (not erased); only tmp_13 is the returning node.
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
def fused_rms_norm_kernel(
    x_ptr,       # tmp_2: already-scaled float32 tensor [N_ROWS, N]
    w_ptr,       # in_1: bfloat16 weight [N]
    out_ptr,     # tmp_13: bfloat16 output [N_ROWS, N]
    N,
    N_ROWS,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program handles ALL rows; weight loaded once
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row_idx * N

    # Load x (float32) — already scaled by in_2
    x = tl.load(x_ptr + row_start + offsets)

    # RMS norm: mean(x^2) + eps -> rsqrt
    x_sq = x * x
    x_sum = tl.sum(x_sq)
    rms_inv = 1.0 / tl.sqrt(x_sum / N + 1e-6)

    # Load weight (bfloat16) and cast to float32
    weight = tl.load(w_ptr + offsets).to(tl.float32)

    # tmp_13 = (x * rms_inv) * (1.0 + weight), cast to bfloat16
    out = (x * rms_inv) * (1.0 + weight)
    tl.store(out_ptr + row_start + offsets, out.to(tl.bfloat16))


@torch.fx.wrap
def fused_rms_norm(tmp_2, in_1):
    N = tmp_2.shape[-1]
    N_ROWS = tmp_2.numel() // N

    out1 = torch.empty_like(tmp_2)   # bfloat16 output (tmp_13)

    fused_rms_norm_kernel[(N_ROWS,)](
        tmp_2, in_1, out1,
        N, N_ROWS,
        BLOCK_SIZE=2048,
        num_warps=8,
    )

    return out1


def replacement_func():
    return fused_rms_norm