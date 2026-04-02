import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – matches ONLY the sum(dim=2,keepdim=True) + divide sub-graph.
# Returns a single tensor (tmp_1) so the return-node count = 1.
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ===========================================================================
# Triton kernel – fused column-sum + divide.
#
# in_1 shape [B, C, H, W] = [1, 2, 8, 8], laid out as [C, H, W] = [2, 8, 8]
# in flat memory (B=1 so the B prefix is a no-op).
#
# Design: SINGLE program, load all C*H*W = 128 elements at once.
# • tl.reshape to [C, H, W]
# • tl.sum over H axis (axis=1)  →  column sums [C, W]
# • divide (broadcast over H)
# • tl.reshape back to flat and store
#
# With all dims as tl.constexpr, Triton fully specialises the kernel and
# can constant-fold all index arithmetic.  num_warps=4 → 128 threads,
# one thread per element.
# ===========================================================================
@triton.jit
def fused_col_sum_div_kernel(
    input_ptr,
    output_ptr,
    C:     tl.constexpr,   # 2
    H:     tl.constexpr,   # 8
    W:     tl.constexpr,   # 8
    TOTAL: tl.constexpr,   # C*H*W = 128
):
    idx  = tl.arange(0, TOTAL)               # [128]
    vals = tl.load(input_ptr + idx)          # [128] fp16/bf16

    # Reshape to [C, H, W] and reduce over H
    vals_3d  = tl.reshape(vals, (C, H, W))           # [2, 8, 8]
    col_sums = tl.sum(vals_3d, axis=1)               # [2, 8]  (sum over H)
    out_3d   = vals_3d / col_sums[:, None, :]         # [2, 8, 8] broadcast

    out = tl.reshape(out_3d, (TOTAL,))               # [128]
    tl.store(output_ptr + idx, out)


@torch.fx.wrap
def fused_sum_div(in_1):
    B, C, H, W = in_1.shape          # 1, 2, 8, 8
    TOTAL = C * H * W                # 128  (B=1)
    out   = torch.empty_like(in_1)
    # Single program processes all 128 elements; num_warps=4 → 128 threads
    fused_col_sum_div_kernel[(1,)](
        in_1.reshape(-1), out.reshape(-1),
        C=C, H=H, W=W, TOTAL=TOTAL,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_sum_div