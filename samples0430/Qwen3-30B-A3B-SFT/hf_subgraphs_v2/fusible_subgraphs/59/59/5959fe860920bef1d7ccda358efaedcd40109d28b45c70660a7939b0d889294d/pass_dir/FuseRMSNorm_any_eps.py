import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: rms_norm with eps as a FREE PARAMETER so that it matches graphs
# using EITHER 1e-06 (SmolLM3-3B) OR 1e-05 (TinyLlama) as the epsilon value.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_2, eps):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + eps
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2, eps):
    return (in_0, in_2)


# ──────────────────────────────────────────────────────────────────────────────
# Shared Triton kernel (identical to FuseRMSNorm_scale_weight_bf16.py)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _rms_norm_shared_kernel(
    x_ptr,          # bfloat16 input  [N, D]
    w_ptr,          # bfloat16 weight [D]
    out_ptr,        # bfloat16 output [N, D]
    N,              # number of rows
    D,              # row length (last dim, typically 2048)
    eps,            # RMSNorm epsilon (runtime scalar)
    BLOCK_D: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_BLOCK
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # Load weight once, reused across all ROWS_PER_BLOCK rows
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for i in tl.static_range(ROWS_PER_BLOCK):
        row_id = base_row + i
        row_mask = mask & (row_id < N)

        x_raw = tl.load(x_ptr + row_id * D + cols, mask=row_mask, other=0.0)
        x = x_raw.to(tl.float32)

        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / D
        norm_factor = tl.rsqrt(mean_sq + eps)

        out = (x * norm_factor) * w
        tl.store(out_ptr + row_id * D + cols, out.to(tl.bfloat16), mask=row_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper  (identical across both pass files for the limit check)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def rms_norm_shared_dispatch(in_0, in_2):
    D = in_2.shape[-1]
    N = in_2.numel() // D
    out = torch.empty_like(in_2)
    BLOCK_D = 2048
    ROWS_PER_BLOCK = 4
    grid = ((N + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,)
    _rms_norm_shared_kernel[grid](
        in_2, in_0, out, N, D, 1e-6,
        BLOCK_D=BLOCK_D, ROWS_PER_BLOCK=ROWS_PER_BLOCK, num_warps=8,
    )
    return out


def replacement_func():
    return rms_norm_shared_dispatch