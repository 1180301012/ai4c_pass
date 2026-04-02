"""
Fused pass: gelu + transpose(1,2) + add
Works for ALL three target graphs (bfloat16/dropout=0.05, float16/dropout=0.1,
float32/dropout=0.1) because the pattern stops BEFORE dropout.

KEY OPTIMIZATION — 2D tiled kernel with coalesced memory access:
  • tmp4 [1,C,T]:  inner dim = T (stride=1) → load (BC,BT) tiles, T is fast dim → COALESCED
  • in3  [1,T,C]:  inner dim = C (stride=1) → load (BT,BC) tiles, C is fast dim → COALESCED
  • out  [1,T,C]:  inner dim = C (stride=1) → store (BT,BC) tiles, C is fast dim → COALESCED
  • tl.trans() transposes the gelu result from (BC,BT)→(BT,BC) in registers

Grid: (ceil(T/BT), ceil(C/BC)), autotuned over BT and BC.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  — matches gelu + transpose(1,2) + add across ALL three graphs
# ---------------------------------------------------------------------------
def pattern(tmp_4, in_3):
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    return tmp_7


def replacement_args(tmp_4, in_3):
    return (tmp_4, in_3)


# ---------------------------------------------------------------------------
# Triton kernel  — 2D tiled, coalesced, autotuned
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # BT=32 configs — optimal for FP32 (32×4=128B = 1 cache line in T dim)
        triton.Config({'BT': 32, 'BC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BT': 32, 'BC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BT': 32, 'BC': 128}, num_warps=4, num_stages=3),
        triton.Config({'BT': 32, 'BC':  64}, num_warps=4, num_stages=2),
        triton.Config({'BT': 32, 'BC':  64}, num_warps=8, num_stages=2),
        triton.Config({'BT': 32, 'BC': 256}, num_warps=4, num_stages=2),
        triton.Config({'BT': 32, 'BC': 256}, num_warps=8, num_stages=2),
        # BT=64 configs — optimal for BF16/FP16 (64×2=128B = 1 cache line in T dim)
        triton.Config({'BT': 64, 'BC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BT': 64, 'BC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BT': 64, 'BC':  64}, num_warps=4, num_stages=2),
        triton.Config({'BT': 64, 'BC':  64}, num_warps=8, num_stages=2),
        triton.Config({'BT': 64, 'BC': 256}, num_warps=4, num_stages=2),
        # BT=16 configs — high-parallelism (many programs for good SM utilization)
        triton.Config({'BT': 16, 'BC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BT': 16, 'BC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BT': 16, 'BC': 128}, num_warps=4, num_stages=3),
        triton.Config({'BT': 16, 'BC':  64}, num_warps=4, num_stages=2),
        triton.Config({'BT': 16, 'BC':  64}, num_warps=8, num_stages=2),
        triton.Config({'BT': 16, 'BC': 256}, num_warps=4, num_stages=2),
        triton.Config({'BT': 16, 'BC': 256}, num_warps=8, num_stages=2),
        # BT=8 configs — maximum parallelism
        triton.Config({'BT':  8, 'BC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BT':  8, 'BC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BT':  8, 'BC': 256}, num_warps=4, num_stages=2),
        # BT=32 num_warps=8
        triton.Config({'BT': 32, 'BC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BT': 32, 'BC':  64}, num_warps=8, num_stages=2),
        triton.Config({'BT': 64, 'BC':  64}, num_warps=8, num_stages=2),
    ],
    # KEY IMPROVEMENT: include DTYPE_NBYTES so BF16/FP16 and FP32 get
    # independently tuned configs (otherwise all 3 graphs share the same
    # cached config because they all have T=249, C=1024)
    key=['T', 'C', 'DTYPE_NBYTES'],
)
@triton.jit
def _fused_gelu_add_kernel_2d(
    # tmp4 : [1, C, T_orig]   stride_c=T_orig≈250, stride_t=1
    tmp4_ptr,
    tmp4_stride_c,
    tmp4_stride_t,
    # in3  : [1, T, C]  contiguous  stride_t=C=1024, stride_c=1
    in3_ptr,
    in3_stride_t,
    # out  : [1, T, C]  contiguous  stride_t=C=1024, stride_c=1
    out_ptr,
    out_stride_t,
    # Dimensions
    T, C,
    DTYPE_NBYTES: tl.constexpr,  # element size: 2 for bf16/fp16, 4 for fp32 (autotune key)
    BT: tl.constexpr,   # tile size in time dimension
    BC: tl.constexpr,   # tile size in channel dimension
):
    """
    2D tiled kernel. One program per (BT-block, BC-block) of the (T, C) output.
    DTYPE_NBYTES is only used as an autotune cache key (not in computation).
    """
    pid_t = tl.program_id(0)   # index along T-block axis
    pid_c = tl.program_id(1)   # index along C-block axis

    t_start = pid_t * BT
    c_start = pid_c * BC

    offs_t = t_start + tl.arange(0, BT)   # [BT]
    offs_c = c_start + tl.arange(0, BC)   # [BC]

    mask_t = offs_t < T     # [BT]
    mask_c = offs_c < C     # [BC]

    # ---- Load tmp4[0, c_start:c_start+BC, t_start:t_start+BT] → shape (BC, BT) ----
    # Fast dim = T (stride_t=1) → coalesced
    mask_ct = mask_c[:, None] & mask_t[None, :]   # (BC, BT)
    raw = tl.load(
        tmp4_ptr
        + offs_c[:, None] * tmp4_stride_c
        + offs_t[None, :] * tmp4_stride_t,
        mask=mask_ct,
        other=0.0,
    )  # shape (BC, BT), original dtype

    # ---- GELU in float32 → shape (BC, BT) ----
    x = raw.to(tl.float32)
    # Exact erf-based GELU for all dtypes (fp32 gets zero numerical error)
    INV_SQRT2 = 0.7071067811865476
    gelu_ct = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))

    # ---- Transpose gelu: (BC, BT) → (BT, BC) in registers ----
    gelu_tc = tl.trans(gelu_ct)   # (BT, BC)

    # ---- Load in3[0, t_start:t_start+BT, c_start:c_start+BC] → shape (BT, BC) ----
    # Fast dim = C (stride_c=1) → coalesced
    mask_tc = mask_t[:, None] & mask_c[None, :]   # (BT, BC)
    y = tl.load(
        in3_ptr
        + offs_t[:, None] * in3_stride_t
        + offs_c[None, :],
        mask=mask_tc,
        other=0.0,
    ).to(tl.float32)   # (BT, BC)

    # ---- Residual add ----
    out_tc = gelu_tc + y   # (BT, BC)

    # ---- Store → output[0, t_start:t_start+BT, c_start:c_start+BC] ----
    tl.store(
        out_ptr
        + offs_t[:, None] * out_stride_t
        + offs_c[None, :],
        out_tc.to(raw.dtype),
        mask=mask_tc,
    )


# ---------------------------------------------------------------------------
# Wrapper  (@torch.fx.wrap → single opaque FX node, single tensor return)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_gelu_add(tmp_4, in_3):
    """
    tmp_4 : [1, C, T]  conv1d slice (strided in C dim, stride ≈ 250)
    in_3  : [1, T, C]  residual hidden states (contiguous)
    Returns: [1, T, C]  = gelu(tmp_4.T) + in_3   (= tmp_7 in the original graph)
    """
    _B, C, T = tmp_4.shape       # [1, 1024, 249]

    out = torch.empty(1, T, C, dtype=tmp_4.dtype, device=tmp_4.device)

    # 2D grid: axis-0 = T blocks, axis-1 = C blocks
    def grid(meta):
        return (
            triton.cdiv(T, meta['BT']),
            triton.cdiv(C, meta['BC']),
        )

    _fused_gelu_add_kernel_2d[grid](
        tmp_4,
        tmp_4.stride(1),   # stride in C (≈ 250 for sliced conv view)
        tmp_4.stride(2),   # stride in T (= 1)
        in_3,
        in_3.stride(1),    # stride in T (= C = 1024)
        out,
        out.stride(1),     # stride in T (= C = 1024)
        T, C,
        DTYPE_NBYTES=tmp_4.element_size(),  # 2 for bf16/fp16, 4 for fp32
    )

    return out


# ---------------------------------------------------------------------------
# Replacement entry point  (zero-arg, returns callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_gelu_add