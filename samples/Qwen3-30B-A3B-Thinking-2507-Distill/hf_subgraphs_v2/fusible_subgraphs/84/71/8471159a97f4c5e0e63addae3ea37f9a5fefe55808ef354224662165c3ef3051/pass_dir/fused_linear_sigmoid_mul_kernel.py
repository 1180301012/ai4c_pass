import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small BLOCK_HW: good for HW=3136 where 3136=64*49 (no waste)
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def fused_linear_sigmoid_mul_kernel(
    in2_ptr,     # [B, K]
    in1_ptr,     # [C, K]
    in0_ptr,     # [C]
    in3_ptr,     # [B, C, HW]
    out_ptr,     # [B, C, HW]
    B, C, K, HW,
    BLOCK_K:  tl.constexpr,   # K rounded up to next power-of-2 (always 8)
    BLOCK_HW: tl.constexpr,   # autotuned spatial tile
):
    # 2-D grid: axis0 = bc_idx (b*C+c), axis1 = hw_block
    bc_idx   = tl.program_id(0)
    hw_block = tl.program_id(1)

    b = bc_idx // C
    c = bc_idx  % C

    # ── 1. linear(b, c) = dot(in2[b,:], in1[c,:]) + in0[c] ────────────────
    k_offs  = tl.arange(0, BLOCK_K)
    mask_k  = k_offs < K
    in2_val = tl.load(in2_ptr + b * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
    in1_val = tl.load(in1_ptr + c * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
    linear_val = tl.sum(in2_val * in1_val, axis=0) + tl.load(in0_ptr + c).to(tl.float32)

    # ── 2. Sigmoid ─────────────────────────────────────────────────────────
    scale = tl.sigmoid(linear_val)

    # ── 3. Apply scale to in3[b, c, hw_tile] ───────────────────────────────
    hw_start   = hw_block * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw    = hw_offsets < HW

    base     = b * C * HW + c * HW
    in3_vals = tl.load(in3_ptr + base + hw_offsets, mask=mask_hw, other=0.0).to(tl.float32)
    out_vals = in3_vals * scale
    # tl.store auto-converts float32 → output dtype (bf16/fp16/fp32)
    tl.store(out_ptr + base + hw_offsets, out_vals, mask=mask_hw)


@torch.fx.wrap
def fused_linear_sigmoid_mul(in_0, in_1, in_2, in_3):
    """
    Fused kernel for:
        linear = F.linear(in_2, in_1, in_0)   # [B, C]
        scale  = sigmoid(linear).view(B, C, 1, 1)
        out    = in_3 * scale                  # [B, C, H, W]
    """
    B, C, H, W = in_3.shape
    K  = in_2.shape[-1]    # 8
    HW = H * W
    out = torch.empty_like(in_3)

    def grid(meta):
        return (B * C, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_linear_sigmoid_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, C, K, HW,
        BLOCK_K=8,
    )
    return out