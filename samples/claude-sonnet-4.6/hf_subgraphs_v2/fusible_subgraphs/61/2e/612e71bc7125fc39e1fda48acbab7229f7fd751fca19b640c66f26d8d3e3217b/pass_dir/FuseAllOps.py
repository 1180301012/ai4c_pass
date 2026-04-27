import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Combined pattern: matches the ENTIRE forward computation:
#   tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
#   tmp_1 = in_1 / tmp_0
#   tmp_2 = in_0.t()
#   tmp_3 = tmp_2.to(device(type='cuda'))
#   return (tmp_1, tmp_3)
#
# By matching everything at once we replace all 4 ops with a SINGLE kernel
# launch, minimising both kernel-launch overhead and FX-graph node overhead.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_1, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: handles BOTH outputs in one launch.
#
#   pid in [0, N1):    row-wise L2 normalisation of in_1
#   pid == N1:         flat copy of in_0  ([1,D] → contiguous [D,1])
#
# BLOCK_D chosen at call time as next_power_of_2(D).
# num_warps=1 (32 threads) → minimal scheduling overhead for tiny N.
# ---------------------------------------------------------------------------
@triton.jit
def l2_norm_and_transpose_kernel(
    in1_ptr,
    in0_ptr,
    out1_ptr,
    out0_ptr,
    N1,
    D,
    stride_n1,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    if pid < N1:
        # --- L2-normalise one row of in_1 ---
        x = tl.load(
            in1_ptr + pid * stride_n1 + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inv_norm = tl.rsqrt(tl.sum(x * x, axis=0))
        out = (x * inv_norm).to(tl.bfloat16)
        tl.store(out1_ptr + pid * stride_n1 + cols, out, mask=mask)
    else:
        # --- Flat copy: in_0 [1,D] → out_0 [D,1]
        # Both layouts are D contiguous elements so a simple copy suffices.
        x = tl.load(in0_ptr + cols, mask=mask, other=0.0)
        tl.store(out0_ptr + cols, x, mask=mask)


@torch.fx.wrap
def combined_triton(in_0, in_1):
    # in_1 : [N1, D] bfloat16 → out_1 : [N1, D] bfloat16 (L2-normalised)
    # in_0 : [1,  D] bfloat16 → out_0 : [D,  1] bfloat16 (transposed)
    N1, D = in_1.shape
    out_1 = torch.empty_like(in_1)

    M, D0 = in_0.shape  # M == 1
    out_0 = torch.empty((D0, M), dtype=in_0.dtype, device=in_0.device)

    BLOCK_D = triton.next_power_of_2(D)

    # N1 + 1 programs: first N1 normalise rows, last one copies in_0
    l2_norm_and_transpose_kernel[(N1 + 1,)](
        in_1,
        in_0,
        out_1,
        out_0,
        N1,
        D,
        D,            # stride_n1 = D (row-major)
        BLOCK_D=BLOCK_D,
        num_warps=1,
    )
    return out_1, out_0


def replacement_func():
    return combined_triton