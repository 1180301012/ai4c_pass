import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse matmul(in_2, in_1) * in_0  →  returns tmp_1 only.
#   The subsequent .T stays in the graph and is applied automatically.
#
#   in_0  : scalar tensor (shape=[])
#   in_1  : [K, 1]   e.g. [512, 1]
#   in_2  : [M, K]   e.g. [2, 512]
#   return: tmp_1 = matmul * scale,  shape [M, 1]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel — single program (grid=1), 1D reduction.
#
# Design:
#  • Single thread block (grid=1) handles all M=2 rows sequentially.
#  • M is tl.constexpr → inner loop fully unrolled at compile time.
#  • in_1 and scale are loaded ONCE, reused for all rows.
#  • fp32 accumulation; result cast to output dtype on store.
#  • num_warps=4 (128 threads, 4 fp16 elems/thread) — best config found.
# ---------------------------------------------------------------------------

@triton.jit
def fused_matmul_scale_kernel(
    in_2_ptr,           # [M, K]  row-major, contiguous
    in_1_ptr,           # [K, 1]  contiguous; element [k,0] at offset k
    in_0_ptr,           # scalar tensor
    out_ptr,            # [M]     output (same dtype as inputs)
    K,
    M:       tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k_offsets = tl.arange(0, BLOCK_K)
    mask_k    = k_offsets < K

    b     = tl.load(in_1_ptr + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
    scale = tl.load(in_0_ptr).to(tl.float32)
    out_dtype = out_ptr.dtype.element_ty

    for m in range(M):
        a   = tl.load(in_2_ptr + m * K + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
        dot = tl.sum(a * b, axis=0)
        tl.store(out_ptr + m, (dot * scale).to(out_dtype))


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    """
    Fused: tmp_1 = matmul(in_2, in_1) * in_0
    Single Triton kernel; loads in_1 once; fp32 accumulation; direct dtype.
    Returns tmp_1 with shape [M, 1].
    """
    M = in_2.shape[0]
    K = in_2.shape[1]
    out = torch.empty(M, dtype=in_2.dtype, device=in_2.device)

    fused_matmul_scale_kernel[(1,)](
        in_2, in_1, in_0, out,
        K,
        M=M,
        BLOCK_K=512,
        num_warps=4,
    )

    return out.view(M, 1)


def replacement_func():
    return fused_matmul_scale