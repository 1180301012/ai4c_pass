"""
Fuse torch.matmul(in_1, in_0) + torch.reshape(result, [-1, 128])
Covers: Finnish-NLP convbert-base-generator-finnish (in_0=[38,9,1], in_1=[38,64,9], BM=2432)
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 128])
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel  –  batched matrix-vector product
#   in_1 : [B, M, K]   (left  operand)
#   in_0 : [B, K]      (right operand, squeezed from [B, K, 1])
#   out  : [B*M]  flat; caller reshapes to [-1, 128]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32},  num_warps=1),
        triton.Config({'BLOCK_M': 64},  num_warps=2),
        triton.Config({'BLOCK_M': 128}, num_warps=2),
        triton.Config({'BLOCK_M': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 1024}, num_warps=8),
    ],
    key=['BM', 'M', 'K'],
)
@triton.jit
def _batched_matvec_n128(
    in1_ptr,   # [B, M, K]  – row-major, strides (M*K, K, 1)
    in0_ptr,   # [B, K]     – row-major, strides (K, 1)
    out_ptr,   # [BM]
    BM,        # B * M  (total output elements)
    M,         # inner M dimension
    K,         # reduction dimension (= 9 here)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be >= K; use next power-of-2
):
    pid = tl.program_id(0)
    bm_start = pid * BLOCK_M
    bm_offs  = bm_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    bm_mask  = bm_offs < BM

    b = bm_offs // M    # batch index
    m = bm_offs  % M    # row   index

    k_offs = tl.arange(0, BLOCK_K)   # [BLOCK_K]
    k_mask = k_offs < K

    # Load in_1[b, m, k_offs]
    in1_ptrs = in1_ptr + (b * M * K + m * K)[:, None] + k_offs[None, :]
    in1_vals = tl.load(in1_ptrs,
                       mask=bm_mask[:, None] & k_mask[None, :],
                       other=0.0).to(tl.float32)

    # Load in_0[b, k_offs]
    in0_ptrs = in0_ptr + (b * K)[:, None] + k_offs[None, :]
    in0_vals = tl.load(in0_ptrs,
                       mask=bm_mask[:, None] & k_mask[None, :],
                       other=0.0).to(tl.float32)

    # Dot product along K, accumulate in fp32
    result = tl.sum(in1_vals * in0_vals, axis=1)   # [BLOCK_M] fp32

    tl.store(out_ptr + bm_offs, result, mask=bm_mask)


# ---------------------------------------------------------------------------
# Python wrapper (decorated with @torch.fx.wrap so FX tracing sees it as leaf)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_matmul_reshape_n128(in_0, in_1):
    """
    in_0 : [B, K, 1]
    in_1 : [B, M, K]
    returns : [-1, 128]  (same as matmul + reshape)
    """
    B, M, K = in_1.shape[0], in_1.shape[1], in_1.shape[2]
    BM = B * M

    # Remove the trailing size-1 dimension from in_0
    in_0_sq = in_0.view(B, K)  # [B, K]

    # Output buffer (flat); we'll view it as [-1, 128] at the end
    out_flat = torch.empty(BM, dtype=in_1.dtype, device=in_1.device)

    BLOCK_K = 16   # next power-of-2 >= 9
    grid = lambda meta: (triton.cdiv(BM, meta['BLOCK_M']),)

    _batched_matvec_n128[grid](
        in_1, in_0_sq, out_flat,
        BM, M, K,
        BLOCK_K=BLOCK_K,
    )

    return out_flat.view(-1, 128)


# ---------------------------------------------------------------------------
# Entry point for the pass framework
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_matmul_reshape_n128