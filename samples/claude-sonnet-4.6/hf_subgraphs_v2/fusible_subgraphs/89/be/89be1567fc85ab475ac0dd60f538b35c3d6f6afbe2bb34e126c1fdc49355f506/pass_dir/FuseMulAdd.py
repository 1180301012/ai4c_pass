import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
# Match mul + add (single output [N, K, 2, D]).
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Grid: (N, K) — 2D, no integer division.
# Each (b, k) program reads in_2 ONCE and writes both i=0 and i=1 slices.
# BLOCK_D must equal D (=128) so all D elements are processed in one block.
@triton.jit
def _fused_mul_add_kernel(
    in0_ptr,   # [2, D]
    in1_ptr,   # [1, 1, 2, D]
    in2_ptr,   # [N, K, 1, D]
    out_ptr,   # [N, K, 2, D]
    K,
    D,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)

    # Load in_2[b, k, 0, d] ONCE — serves both slices
    in2_val = tl.load(in2_ptr + b * K * D + k * D + d)

    # ── i = 0 ──────────────────────────────────────────────────────────────
    in1_v0 = tl.load(in1_ptr + d)
    in0_v0 = tl.load(in0_ptr + d)
    tl.store(out_ptr + b * K * 2 * D + k * 2 * D + d,
             in2_val * in1_v0 + in0_v0)

    # ── i = 1 ──────────────────────────────────────────────────────────────
    in1_v1 = tl.load(in1_ptr + D + d)
    in0_v1 = tl.load(in0_ptr + D + d)
    tl.store(out_ptr + b * K * 2 * D + k * 2 * D + D + d,
             in2_val * in1_v1 + in0_v1)


# ── Wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_mul_add(in_0, in_1, in_2):
    N = in_2.shape[0]
    K = in_2.shape[1]
    D = in_2.shape[3]

    out = torch.empty((N, K, 2, D), dtype=in_2.dtype, device=in_2.device)

    _fused_mul_add_kernel[(N, K)](
        in_0, in_1, in_2,
        out,
        K, D,
        BLOCK_D=128,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_mul_add