import torch
import triton
import triton.language as tl


# ── Full-graph pattern: conv2d + sigmoid + broadcast-mul + hardtanh ─────────────
# Matches all four operations in the target subgraph.
#
# All inputs:
#   in_0 → bias   [228]           (cpu)
#   in_1 → weight [228, 19, 1, 1] (cpu)
#   in_2 → features [B, 228, H, W] (cuda)
#   in_3 → SE_in   [B, 19,  1, 1]  (cuda)
# ── ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.sigmoid()
    tmp_4  = in_2 * tmp_3
    tmp_5  = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    # Pass bias, weight, features, and SE input to the kernel
    return (in_0, in_1, in_2, in_3)


# ── Triton kernel ───────────────────────────────────────────────────────────────
# 2D grid: axis-0 = B*C (one program per output channel), axis-1 = HW tiles.
# Stays in registers: no re-read of conv output anywhere.
#
# in3  is [B, 19, 1, 1]: element (b, k) at flat b*19+k.
# Because k = c_idx = gc_idx % C = (b*C+c) % C,  in3 flat = b*19+c = gc_idx ✓
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _fused_se_sigmoid_mul_hardtanh_kernel(
    in2_ptr,   # [B, C, H, W]
    in3_ptr,   # [B, K, 1, 1]  K=19
    w_ptr,     # [C, K, 1, 1]
    b_ptr,     # [C]
    out_ptr,
    C,         # == 228
    HW,        # H * W
    K:        tl.constexpr,   # == 19
    BLOCK_HW: tl.constexpr,
):
    # 3D grid: axis-0=batch, axis-1=channel, axis-2=HW tile
    batch_idx = tl.program_id(0)
    c_idx     = tl.program_id(1)
    hw        = tl.program_id(2)

    # ── GEMV: dot(in3[batch, :], w[c, :]) + bias[c] ──────────────────────────
    k_off = tl.arange(0, 32)   # 32 = 2^ceil(log2(19)) padded with zeros
    in3_v = tl.load(in3_ptr + batch_idx * K + k_off,
                    mask=k_off < K, other=0.0)
    w_v   = tl.load(w_ptr   + c_idx     * K + k_off,
                    mask=k_off < K, other=0.0)
    dot   = tl.sum(in3_v.to(tl.float32) * w_v.to(tl.float32), axis=0)
    bias  = tl.load(b_ptr + c_idx).to(tl.float32)
    scale = tl.sigmoid(dot + bias)

    # ── Fused sigmoid-scale * in2 + clamp ─────────────────────────────────────
    hw_offs = hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW
    base    = (batch_idx * C + c_idx) * HW
    in2_v   = tl.load(in2_ptr + base + hw_offs, mask=mask, other=0.0)
    result  = tl.minimum(tl.maximum(in2_v.to(tl.float32) * scale, 0.0), 6.0)
    tl.store(out_ptr + base + hw_offs, result.to(in2_v.dtype), mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul_hardtanh(in_0, in_1, in_2, in_3):
    # in_0→bias[C], in_1→weight[C,K,1,1], in_2→features[B,C,H,W], in_3→SE_input[B,K,1,1]
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW = H * W
    out = torch.empty_like(in_2)
    grid = lambda meta: (B, C, triton.cdiv(HW, meta['BLOCK_HW']))
    _fused_se_sigmoid_mul_hardtanh_kernel[grid](
        in2_ptr=in_2, in3_ptr=in_3, w_ptr=in_1, b_ptr=in_0, out_ptr=out,
        C=C, HW=HW, K=19,
    )
    return out



def replacement_func():
    return fused_sigmoid_mul_hardtanh