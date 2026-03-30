import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full computation in all three dtype graphs.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused Triton kernel – one program per softmax row.
#
# Shape: in_0 [H=16], in_1/in_2/out [B=1, H=16, R=196, N=196].
# Programs: B×H×R = 3 136.
#
# Precision strategy:
#   • Sigmoid gate: computed in fp32 (exp precision), cast to input dtype.
#   • Softmax: single fp32 buffer for x2 (one conversion, reused for
#     max / shift / exp / normalize).  Avoids the 3× extra conversions that
#     a native-dtype shift would require.
#   • Gating multiply-add: executed in INPUT dtype (float16/bf16/fp32).
#     This halves working-set registers vs an all-fp32 gating path.
#
# Autotune key includes 'elem_bytes' so float16/bfloat16 (2 B/elem) and
# float32 (4 B/elem) get INDEPENDENT autotune caches and can each find their
# own optimal num_warps.  Without this, a shared cache would pick a config
# tuned for one dtype that may be suboptimal for the other.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['H', 'N'],
)
@triton.jit
def fused_gated_softmax_kernel(
    in_0_ptr,           # [H]           gate parameters
    in_1_ptr,           # [B, H, R, N]  patch scores
    in_2_ptr,           # [B, H, R, N]  position scores
    out_ptr,            # [B, H, R, N]  output
    H,                  # 16
    N,                  # 196
    BLOCK_N: tl.constexpr,   # 256 = next pow-2 ≥ N
):
    row_idx  = tl.program_id(0)
    head_idx = (row_idx // N) % H

    # ---- Sigmoid gate (fp32 for exp precision, then cast to input dtype) ----
    gate     = tl.load(in_0_ptr + head_idx, eviction_policy='evict_last').to(tl.float32)
    s_fp32   = 1.0 / (1.0 + tl.exp(-gate))
    oms_fp32 = 1.0 - s_fp32

    row_base = row_idx * N
    cols     = tl.arange(0, BLOCK_N)
    mask     = cols < N

    # ---- Load position scores in native dtype ----
    x2_raw = tl.load(in_2_ptr + row_base + cols, mask=mask, other=float('-inf'),
                      eviction_policy='evict_first')

    # ---- Softmax: single fp32 buffer (one conversion, reused for max/shift/exp) ----
    x2      = x2_raw.to(tl.float32)
    x2_max  = tl.max(x2, axis=0)
    x2      = x2 - x2_max
    x2_exp  = tl.exp(x2)
    x2_sum  = tl.sum(x2_exp, axis=0)
    sm_fp32 = x2_exp / x2_sum

    # ---- Load patch scores in native dtype ----
    x1_raw = tl.load(in_1_ptr + row_base + cols, mask=mask, other=0.0,
                      eviction_policy='evict_first')

    # ---- Gating multiply-add in native dtype (saves registers vs all-fp32) ----
    sm  = sm_fp32.to(x1_raw.dtype)
    s   = s_fp32.to(x1_raw.dtype)
    oms = oms_fp32.to(x1_raw.dtype)
    out = oms * x1_raw + s * sm
    tl.store(out_ptr + row_base + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    B, H, R, N = in_2.shape      # [1, 16, 196, 196]
    total_rows  = B * H * R       # 3 136

    # in_0 may live on CPU; .to() is a no-op if already on CUDA.
    in_0_dev = in_0.to(in_1.device)

    out = torch.empty_like(in_1)

    # Pass element size so autotune picks dtype-specific configs.
    fused_gated_softmax_kernel[(total_rows,)](
        in_0_dev,
        in_1,
        in_2,
        out,
        H,
        N,
        BLOCK_N=256,
    )

    return out


def replacement_func():
    return kernel_wrapper