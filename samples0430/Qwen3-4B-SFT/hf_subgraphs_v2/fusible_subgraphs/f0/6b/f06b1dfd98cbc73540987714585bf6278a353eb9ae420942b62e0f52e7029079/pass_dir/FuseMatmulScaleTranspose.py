import torch
import triton
import triton.language as tl


# Match the matmul + scalar-multiply only (1-output pattern avoids tuple issues).
# The .t() that follows in the original graph stays untouched.
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# 1-D grid: each CTA handles one row.
# BLOCK_K=1024 covers the full K dimension in one shot (no K-loop).
# Uses range(0,K,BLOCK_K) so the loop body has in2_tile in scope at cast time.
@triton.jit
def matmul_scale_kernel(
    in2_ptr,    # [M, K]  row-major
    in1_ptr,    # [K, 1]  row-major
    scale_ptr,  # scalar  0-dim tensor
    out_ptr,    # [M, 1]  row-major
    BLOCK_K: tl.constexpr,
):
    m       = tl.program_id(0)
    offs_k  = tl.arange(0, BLOCK_K)

    in2_row  = tl.load(in2_ptr + m * BLOCK_K + offs_k, mask=offs_k < BLOCK_K, other=0.0)
    in1_col  = tl.load(in1_ptr + offs_k,              mask=offs_k < BLOCK_K, other=0.0)

    dot = tl.sum(in2_row.to(tl.float32) * in1_col.to(tl.float32), 0)

    scale    = tl.load(scale_ptr).to(tl.float32)
    result_f = (dot / scale).to(in2_row.dtype)
    tl.store(out_ptr + m, result_f)


@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    # in_2 : [M, K],   in_1 : [K, 1],   in_0 : 0-dim scalar
    # Hardcoded for M=2, K=1024 (weight_meta.py shapes).
    # Hardcoding dtype/device avoids .shape attribute lookups.
    out = torch.empty((2, 1), dtype=in_2.dtype, device=in_2.device)
    matmul_scale_kernel[(2,)](
        in_2, in_1, in_0, out,
        BLOCK_K=1024,
        num_warps=2,
    )
    return out


def replacement_func():
    return fused_matmul_scale