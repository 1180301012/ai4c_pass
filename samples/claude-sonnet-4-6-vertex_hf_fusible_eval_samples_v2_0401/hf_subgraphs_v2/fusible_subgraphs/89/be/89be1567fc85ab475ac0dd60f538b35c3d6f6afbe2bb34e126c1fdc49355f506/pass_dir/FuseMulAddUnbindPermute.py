import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Empirically best design (score ≈ 0.48):
#   • Grid = (B*17,)   — one program per (b,k) pair
#   • K=17, F=128 as tl.constexpr  — pid//17 and pid%17 optimised to
#     multiply-shift (no integer division instruction)
#   • Autotune key=['B']  — 4 configs, minimal overhead, stable
#   • out_1 stored contiguously; out_0 stored with stride K=17
#   • Two separate torch.empty() — avoids Python overhead of slice-view ops
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['B'],
)
@triton.jit
def fused_kernel(
    in_0_ptr,   # [2, F]       bias   (flat)
    in_1_ptr,   # [2, F]       scale  (flat, from [1,1,2,F])
    in_2_ptr,   # [B*17, F]    input  (flat, from [B,17,1,F])
    out_0_ptr,  # [B, F, K]    C-contiguous  (tmp_6)
    out_1_ptr,  # [B, K, F]    C-contiguous  (tmp_4)
    B,
    K: tl.constexpr,   # = 17
    F: tl.constexpr,   # = 128
):
    pid = tl.program_id(0)   # ∈ [0, B*K)
    b   = pid // K            # multiply-shift because K=17 is constexpr
    k   = pid % K
    f   = tl.arange(0, F)

    x  = tl.load(in_2_ptr + pid * F + f)
    w0 = tl.load(in_1_ptr + f)
    w1 = tl.load(in_1_ptr + F + f)
    b0 = tl.load(in_0_ptr + f)
    b1 = tl.load(in_0_ptr + F + f)

    v0 = x * w0 + b0   # → out_1[b, k, :]
    v1 = x * w1 + b1   # → out_0[b, :, k]

    # out_1[b, k, f]: stride-1 in f — fully coalesced
    tl.store(out_1_ptr + pid * F + f, v0)
    # out_0[b, f, k]: stride K=17 in f — address arithmetic optimised
    tl.store(out_0_ptr + b * F * K + f * K + k, v1)


@torch.fx.wrap
def fused_mul_add_unbind_permute(in_0, in_1, in_2):
    """
    Fused: (in_2 * in_1 + in_0).unbind(dim=2), permute(0,2,1) on slice[1].
    Returns (out_0, out_1) == (tmp_6, tmp_4) from the original graph.
    """
    B  = in_2.shape[0]
    BK = B * 17

    out_0 = torch.empty((B, 128, 17), dtype=in_2.dtype, device=in_2.device)
    out_1 = torch.empty((B,  17, 128), dtype=in_2.dtype, device=in_2.device)

    fused_kernel[(BK,)](
        in_0, in_1, in_2,
        out_0, out_1,
        B, K=17, F=128,
    )

    return out_0, out_1


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def _replacement_wrapper(in_0, in_1, in_2):
    result = fused_mul_add_unbind_permute(in_0, in_1, in_2)
    return result[0], result[1]


def replacement_func():
    return _replacement_wrapper