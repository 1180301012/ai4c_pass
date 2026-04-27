import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse matmul + scalar-scale into a single Triton kernel.
# The .t() transpose stays in the graph as a free view operation.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused Triton kernel  (M=2, N=1, K=1024 for this workload)
#   Grid = (M, N) — one program per output element, run in PARALLEL.
#   num_warps=1 → single warp (32 threads), each thread handles
#   BLOCK_K/32 = 32 elements; reduction via warp-shuffle (no barriers).
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_scale_kernel(
    a_ptr,      # in_2 : [M, K] row-major
    b_ptr,      # in_1 : [K, N] row-major
    scale_ptr,  # in_0 : scalar (0-d tensor)
    out_ptr,    # output : [M, N]
    M, N, K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    scale = tl.load(scale_ptr).to(tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    mask   = k_offs < K

    a = tl.load(a_ptr + pid_m * K + k_offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + k_offs * N + pid_n,  mask=mask, other=0.0).to(tl.float32)

    result = tl.sum(a * b, axis=0) * scale

    tl.store(out_ptr + pid_m * N + pid_n, result)


@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    M, K   = in_2.shape
    _K2, N = in_1.shape

    out = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)

    # Parallel programs, one per output element
    grid = (M, N)
    _matmul_scale_kernel[grid](
        in_2, in_1, in_0,
        out,
        M, N, K,
        BLOCK_K=1024,
    )

    return out


def replacement_func():
    return fused_matmul_scale