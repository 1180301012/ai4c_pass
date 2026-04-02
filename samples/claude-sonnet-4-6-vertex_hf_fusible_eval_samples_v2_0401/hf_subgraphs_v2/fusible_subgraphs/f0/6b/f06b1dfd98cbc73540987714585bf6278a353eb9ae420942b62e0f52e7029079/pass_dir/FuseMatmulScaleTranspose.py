import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match   matmul(in_2, in_1) * in_0   →  tmp_1   [M, 1]
#
# The `.t()` (tmp_2) stays in the graph, applied to our output automatically.
# Inputs:   in_0 – scalar logit_scale  []
#           in_1 – column vector       [K, 1]
#           in_2 – row matrix          [M, K]
# Output:   tmp_1 – [M, 1]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: ONE program per OUTPUT ROW (grid = (M,)).
#
#  Each block handles one row of in_2 and the full K dimension in a single
#  pass (BLOCK_K = 1024 = K).  Launching M parallel blocks enables the GPU
#  to execute all M dot-products simultaneously on different SMs.
#  num_warps=4 (128 threads × 8 elements/thread) gives optimal load
#  parallelism for K=1024 with a single cross-warp reduction step.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_matmul_scale_kernel(
    in2_ptr, in1_ptr, in0_ptr,
    out_ptr,
    M, K,
    stride_in2_m, stride_in2_k,
    stride_in1_k,
    stride_out_m,
    BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row = tl.program_id(0)

    k_offs = tl.arange(0, BLOCK_K)
    mask   = k_offs < K

    # Load the full row and column vector in one shot (BLOCK_K >= K)
    a = tl.load(
        in2_ptr + row * stride_in2_m + k_offs * stride_in2_k,
        mask=mask, other=0.0,
    )
    b = tl.load(
        in1_ptr + k_offs * stride_in1_k,
        mask=mask, other=0.0,
    )

    # tl.sum on a 1-D vector returns a scalar directly – no [0] indexing needed
    dot   = tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=0)
    scale = tl.load(in0_ptr).to(tl.float32)

    tl.store(out_ptr + row * stride_out_m, (dot * scale).to(DTYPE))


# ---------------------------------------------------------------------------
# Python wrapper – grid=(M,), returns [M, 1].
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    """
    Fused:  tmp_1 = matmul(in_2, in_1) * in_0   →  [M, 1]
    """
    M = in_2.shape[0]
    K = in_2.shape[1]

    tmp_1 = torch.empty((M, 1), dtype=in_2.dtype, device=in_2.device)

    _fused_matmul_scale_kernel[(M,)](
        in_2, in_1, in_0,
        tmp_1,
        M, K,
        in_2.stride(0),   # stride_in2_m
        in_2.stride(1),   # stride_in2_k
        in_1.stride(0),   # stride_in1_k
        tmp_1.stride(0),  # stride_out_m
        BLOCK_K=1024,
        DTYPE=_DTYPE_MAP[in_2.dtype],
        num_warps=4,
        num_stages=1,
    )

    return tmp_1


def replacement_func():
    return fused_matmul_scale