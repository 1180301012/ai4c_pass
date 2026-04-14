"""
Shared Triton kernels for EncNet_R101 subgraph optimization.

Three dispatch routes via a single @torch.fx.wrap wrapper (6-arg signature):
  route="combined": single Grid(I) launch doing BOTH dist-scale and
                    expand-subtract in one kernel, interleaving HBM reads
                    (in_1) with HBM writes (es_out) for bidirectional BW.
  route="ds" : fallback — dist-scale only  → returns [B,I,K]
  route="es" : fallback — expand-sub only  → returns [B,I,K,F]
"""

import torch
import triton
import triton.language as tl


# =========================================================================
# COMBINED kernel  Grid(I=4096)
# For each spatial pos i:
#   • loads in4[i,:] ONCE (reused for K expand-sub iterations)
#   • loops k=0..K-1:
#       - dist-scale: load in1[i,k,:] (HBM), compute sq-dist, scale, store
#       - expand-sub: load in0[k,:]   (L1),  subtract in4, store (HBM)
# HBM traffic per block: 32KB reads (in1) + 32KB writes (es_out)
# Bidirectional bandwidth utilisation: reads ↔ writes overlap in same block.
# =========================================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["F", "K"],
)
@triton.jit
def _combined_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, in4_ptr,
    ds_out_ptr, es_out_ptr,
    I, K, F,
    stride_in1_i, stride_in1_k, stride_in2_k,
    BLOCK_F: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    i_idx = tl.program_id(0)
    f_offs = tl.arange(0, BLOCK_F)

    # Load in4[i,:] ONCE (tiny, reused for all K expand-sub operations)
    in4_vals = tl.load(in4_ptr + i_idx * F + f_offs)

    for k in range(K):
        # ---- dist-scale: read in1 from HBM, compute, store tiny output ----
        in1v = tl.load(
            in1_ptr + i_idx * stride_in1_i + k * stride_in1_k + f_offs
        ).to(tl.float32)
        in2v = tl.load(in2_ptr + k * stride_in2_k + f_offs).to(tl.float32)
        diff = in1v - in2v
        dist = tl.sum(diff * diff)
        scale = tl.load(in3_ptr + k).to(tl.float32)
        result = dist * scale
        if IS_BF16:
            tl.store(ds_out_ptr + i_idx * K + k, result.to(tl.bfloat16))
        else:
            tl.store(ds_out_ptr + i_idx * K + k, result.to(tl.float16))

        # ---- expand-sub: read in0 from L1-cache, write es_out to HBM ----
        in0v = tl.load(in0_ptr + k * F + f_offs)
        tl.store(es_out_ptr + (i_idx * K + k) * F + f_offs, in4_vals - in0v)


# =========================================================================
# Separate dist-scale kernel  Grid(I,)   — fallback
# =========================================================================
@triton.jit
def _dist_scale_kernel(
    in1_ptr, in2_ptr, in3_ptr, out_ptr,
    I, K, F,
    stride_in1_i, stride_in1_k, stride_in2_k,
    BLOCK_F: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    i_idx = tl.program_id(0)
    f_offs = tl.arange(0, BLOCK_F)
    for k in range(K):
        in1v = tl.load(
            in1_ptr + i_idx * stride_in1_i + k * stride_in1_k + f_offs,
            eviction_policy='evict_first',    # streaming: don't pollute L1
        ).to(tl.float32)
        in2v = tl.load(
            in2_ptr + k * stride_in2_k + f_offs,
            eviction_policy='evict_last',     # small (32KB): keep in L1
        ).to(tl.float32)
        diff = in1v - in2v
        dist = tl.sum(diff * diff)
        scale = tl.load(in3_ptr + k).to(tl.float32)
        result = dist * scale
        if IS_BF16:
            tl.store(out_ptr + i_idx * K + k, result.to(tl.bfloat16))
        else:
            tl.store(out_ptr + i_idx * K + k, result.to(tl.float16))


# =========================================================================
# Separate expand-sub kernel  Grid(I,)   — fallback
# =========================================================================
@triton.jit
def _expand_sub_kernel(
    in0_ptr, in4_ptr, out_ptr,
    I, K, F,
    BLOCK_F: tl.constexpr,
):
    i_idx = tl.program_id(0)
    f_offs = tl.arange(0, BLOCK_F)
    in4_vals = tl.load(in4_ptr + i_idx * F + f_offs,
                       eviction_policy='evict_last')   # reused K times: keep warm
    for k in range(K):
        in0_vals = tl.load(in0_ptr + k * F + f_offs,
                           eviction_policy='evict_last')  # 32KB: keep in L1
        tl.store(out_ptr + (i_idx * K + k) * F + f_offs, in4_vals - in0_vals)


# =========================================================================
# Python-level helpers
# =========================================================================
def _run_combined(in_0, in_1, in_2, in_3, in_4):
    """Single kernel for both distance-scale and expand-subtract."""
    B, I, K, F = in_1.shape
    ds_out = torch.empty((B, I, K), dtype=in_1.dtype, device=in_1.device)
    es_out = torch.empty((B, I, K, F), dtype=in_4.dtype, device=in_4.device)
    _combined_kernel[(I,)](
        in_0, in_1, in_2, in_3, in_4,
        ds_out, es_out,
        I, K, F,
        in_1.stride(1), in_1.stride(2), in_2.stride(2),
        BLOCK_F=F,
        IS_BF16=(in_1.dtype == torch.bfloat16),
    )
    return ds_out, es_out


def _run_distance_scale(in_1, in_2, in_3):
    B, I, K, F = in_1.shape
    out = torch.empty((B, I, K), dtype=in_1.dtype, device=in_1.device)
    _dist_scale_kernel[(I,)](
        in_1, in_2, in_3, out, I, K, F,
        in_1.stride(1), in_1.stride(2), in_2.stride(2),
        BLOCK_F=F,
        IS_BF16=(in_1.dtype == torch.bfloat16),
        num_warps=4,
    )
    return out


def _run_expand_subtract(in_0, in_4):
    K, F = in_0.shape[0], in_0.shape[1]
    I = in_4.shape[1]
    B = 1
    out = torch.empty((B, I, K, F), dtype=in_4.dtype, device=in_4.device)
    _expand_sub_kernel[(I,)](
        in_0, in_4, out, I, K, F,
        BLOCK_F=F,
        num_warps=8,
    )
    return out


# =========================================================================
# Single shared dispatch wrapper – 6 args so all routes fit.
# =========================================================================
@torch.fx.wrap
def shared_fused_kernel(a, b, c, d, e, route):
    """
    Dispatch wrapper shared by ALL pass files.
      route='combined': a=in_0, b=in_1, c=in_2, d=in_3, e=in_4
                        → (ds_out [B,I,K], es_out [B,I,K,F])
      route='ds':       a=in_1, b=in_2,  c=in_3, d,e=dummies
                        → ds_out [B,I,K]
      route='es':       a=in_0, b=in_4,  c,d,e=dummies
                        → es_out [B,I,K,F]
    """
    if route == "combined":
        return _run_combined(a, b, c, d, e)
    elif route == "ds":
        return _run_distance_scale(a, b, c)
    elif route == "es":
        return _run_expand_subtract(a, b)