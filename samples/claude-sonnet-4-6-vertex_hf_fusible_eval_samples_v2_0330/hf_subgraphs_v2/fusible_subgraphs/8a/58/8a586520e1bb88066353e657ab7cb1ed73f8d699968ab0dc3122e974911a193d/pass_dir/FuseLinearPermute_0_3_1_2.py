import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: torch.nn.functional.linear followed by .permute(0, 3, 1, 2)
#
# Shapes (from weight_meta.py):
#   x      (in_3) : [1, 196, 196, 3]   (B, H, W, K)
#   weight (in_1) : [16, 3]             (C, K)
#   bias   (in_0) : [16]                (C,)
#   output         : [1, 16, 196, 196]  (B, C, H, W)  ← written directly
# ─────────────────────────────────────────────────────────────────────────────

def pattern(x, weight, bias):
    linear = torch.nn.functional.linear(x, weight, bias)
    result = linear.permute(0, 3, 1, 2)
    return result


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel
#   Grid : (C, ceil(M / BLOCK_M))   where M = B*H*W
#   Block: computes BLOCK_M output elements for one channel c
#
#   For channel c, hw in [pid_m*BLOCK_M, …):
#     out[c, hw] = sum_k( x_flat[hw, k] * weight[c, k] ) + bias[c]
#
#   Output is stored as [C, M] = [C, B*H*W], which is the same memory layout
#   as a contiguous [B, C, H, W] tensor (B=1), so a simple reshape suffices.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 2048}, num_warps=16, num_stages=3),
    ],
    key=['M'],
)
@triton.jit
def linear_permute_0312_kernel(
    x_ptr,       # [M, 3]  – x flattened, K=3 hardcoded
    weight_ptr,  # [16, 3] – C=16 hardcoded
    bias_ptr,    # [16]
    out_ptr,     # [16, M] – output in [C, M] = permuted layout
    M,           # B * H * W  (runtime)
    BLOCK_M: tl.constexpr,
):
    # 1D grid: each program owns BLOCK_M m-values and processes ALL C=16 channels.
    # This way x is loaded only ONCE per program (not C times).
    pid_m  = tl.program_id(0)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # ── Load x[m, 0], x[m, 1], x[m, 2] once (stride-3 gather, shared across channels) ──
    x0 = tl.load(x_ptr + m_offs * 3 + 0, mask=m_mask, other=0.0)
    x1 = tl.load(x_ptr + m_offs * 3 + 1, mask=m_mask, other=0.0)
    x2 = tl.load(x_ptr + m_offs * 3 + 2, mask=m_mask, other=0.0)

    # ── Iterate over all 16 output channels (Python loop → unrolled at compile time) ──
    for c in range(16):
        # Scalar loads of weight[c, 0:3] and bias[c] – tiny, cached in registers
        w0 = tl.load(weight_ptr + c * 3 + 0)
        w1 = tl.load(weight_ptr + c * 3 + 1)
        w2 = tl.load(weight_ptr + c * 3 + 2)
        b  = tl.load(bias_ptr   + c)

        # Fused multiply-accumulate + bias
        out_c = x0 * w0 + x1 * w1 + x2 * w2 + b   # [BLOCK_M]

        # Coalesced write: output[c, m_offs] (contiguous in m)
        tl.store(out_ptr + c * M + m_offs, out_c, mask=m_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper (must be decorated with @torch.fx.wrap)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_linear_permute_0312(x, weight, bias):
    """
    Fused drop-in for:
        torch.nn.functional.linear(x, weight, bias).permute(0, 3, 1, 2)

    x      : [B, H, W, K]
    weight : [C, K]
    bias   : [C]
    returns: [B, C, H, W]  (contiguous)
    """
    B, H, W, K = x.shape
    C = weight.shape[0]
    M = B * H * W  # = 38 416 for this model

    # Flatten spatial dims: [B, H, W, K] → [M, K]
    x_flat = x.reshape(M, K)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()

    # Weight and bias might live on CPU in the original model spec;
    # move them to the same device as x (no-op if already there).
    weight_dev = weight.to(device=x.device, dtype=x.dtype)
    bias_dev   = bias.to(device=x.device,   dtype=x.dtype)

    # Allocate output in [C, M] layout == contiguous [B, C, H, W] for B=1
    out = torch.empty((C, M), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    linear_permute_0312_kernel[grid](
        x_flat, weight_dev, bias_dev, out,
        M,
    )

    # Reshape [C, M] → [B, C, H, W]
    return out.reshape(B, C, H, W)


def replacement_func():
    return fused_linear_permute_0312