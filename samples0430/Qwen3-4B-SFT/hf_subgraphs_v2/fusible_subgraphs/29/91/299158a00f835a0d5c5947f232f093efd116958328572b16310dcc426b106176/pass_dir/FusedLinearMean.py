import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: Mean along dim -2 (axis=1) of a 3-D tensor [B_rows, S, N]
#   Output: [B_rows, N]
#
# Grid: (B_rows, 1)
# BLOCK_N ≥ N → one tile covers all N columns; no atomic_add; fully correct.
# S=49 constexpr → loop unrolled at trace time (S is compile-time constant).
# No num_stages → compiler chooses the best pipeline depth automatically.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_mean_kernel(
    inp_ptr,    # [B_rows * S, N]  (contiguous row-major)
    out_ptr,    # [B_rows, N]
    N,          # = 448  (runtime stride factor)
    S:        tl.constexpr,   # = 49  constexpr → compiler unrolls the loop
    BLOCK_N:  tl.constexpr,   # >= N (e.g. 512)
):
    pid_b = tl.program_id(0)   # batch row

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N        # mask wasted threads when BLOCK_N > N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # stride_row = S * N: between consecutive batch rows in memory
    row_base = pid_b * S * N
    for s_off in range(S):   # constexpr → Triton unrolls 49 iterations
        inp = tl.load(inp_ptr + row_base + s_off * N + offs_n, mask=mask_n, other=0.0)
        acc += inp.to(tl.float32)

    tl.store(out_ptr + pid_b * N + offs_n, acc * (1.0 / S), mask=mask_n)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def mean_last2_torch(in_3):
    """in_3: [B, 49, 448]  →  out: [B, 448]"""
    B = in_3.shape[0]
    N = in_3.shape[2]   # 448
    S = in_3.shape[1]   # 49

    mean_out = torch.empty((B, N), dtype=in_3.dtype, device=in_3.device)

    # BLOCK_N=512 ≥ N=448: grid = (B, 1), no atomic conflicts
    BLOCK_N = 512
    grid_mean = (B, triton.cdiv(N, BLOCK_N))

    fused_mean_kernel[grid_mean](
        in_3, mean_out,
        N,
        S,
        BLOCK_N,
        num_warps=4,
    )
    return mean_out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern / replacement_args / replacement_func
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_3):
    tmp_3 = in_3.mean(-2)
    return tmp_3


def replacement_args(in_3):
    return (in_3,)


def replacement_func():
    return mean_last2_torch