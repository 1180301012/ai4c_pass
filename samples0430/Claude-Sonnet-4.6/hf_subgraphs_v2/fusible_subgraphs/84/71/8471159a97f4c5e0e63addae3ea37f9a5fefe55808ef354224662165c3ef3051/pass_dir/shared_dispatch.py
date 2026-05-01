import torch
import triton
import triton.language as tl


# ── B=1 specialised kernel (no autotune, fixed config, b=0 arithmetic removed) ──

@triton.jit
def _lss_kernel_b1(
    bias_ptr,    # [64]
    weight_ptr,  # [64, 8]
    input_ptr,   # [1, 8]
    feat_ptr,    # [1, 64, H, W]
    out_ptr,     # [1, 64, H, W]
    HW,
    BLOCK_HW: tl.constexpr,
):
    # b=0 always for B=1 → omit all b-dependent arithmetic
    c      = tl.program_id(0)   # channel index in [0, 64)
    pid_hw = tl.program_id(1)

    # Compute scale = sigmoid(dot(input[0], weight[c]) + bias[c]) in fp32
    bias = tl.load(bias_ptr + c).to(tl.float32)
    k_off = tl.arange(0, 8)
    x_v = tl.load(input_ptr + k_off).to(tl.float32)       # b=0, offset=0
    w_v = tl.load(weight_ptr + c * 8 + k_off).to(tl.float32)
    scale = tl.sigmoid(tl.sum(x_v * w_v, axis=0) + bias)

    # Load, scale, and store feature-map tile (b=0 → base = c * HW)
    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW
    vals    = tl.load(feat_ptr + c * HW + hw_off, mask=hw_mask, other=0.0)
    tl.store(out_ptr  + c * HW + hw_off, vals * scale.to(vals.dtype), mask=hw_mask)


# ── Fixed-config kernel for B>1 small batches (no autotune overhead) ──────────

@triton.jit
def _lss_kernel_fixed(
    bias_ptr,    # [64]
    weight_ptr,  # [64, 8]
    input_ptr,   # [B, 8]
    feat_ptr,    # [B, 64, H, W]
    out_ptr,     # [B, 64, H, W]
    B, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    b = pid_bc >> 6     # pid_bc // 64
    c = pid_bc & 63     # pid_bc %  64

    bias = tl.load(bias_ptr + c).to(tl.float32)
    k_off = tl.arange(0, 8)
    x_v = tl.load(input_ptr + b * 8 + k_off).to(tl.float32)
    w_v = tl.load(weight_ptr + c * 8 + k_off).to(tl.float32)
    scale = tl.sigmoid(tl.sum(x_v * w_v, axis=0) + bias)

    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW
    base    = b * (64 * HW) + c * HW
    vals    = tl.load(feat_ptr + base + hw_off, mask=hw_mask, other=0.0)
    tl.store(out_ptr + base + hw_off, vals * scale.to(vals.dtype), mask=hw_mask)


# ── Autotuned kernel for large batches (B≥64) ─────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=4),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW', 'B'],
)
@triton.jit
def _lss_kernel_shared(
    bias_ptr,    # [64]
    weight_ptr,  # [64, 8]
    input_ptr,   # [B, 8]
    feat_ptr,    # [B, 64, H, W]
    out_ptr,     # [B, 64, H, W]
    B, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    b = pid_bc >> 6
    c = pid_bc & 63

    bias = tl.load(bias_ptr + c).to(tl.float32)
    k_off = tl.arange(0, 8)
    x_v = tl.load(input_ptr + b * 8 + k_off).to(tl.float32)
    w_v = tl.load(weight_ptr + c * 8 + k_off).to(tl.float32)
    scale = tl.sigmoid(tl.sum(x_v * w_v, axis=0) + bias)

    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW
    base    = b * (64 * HW) + c * HW
    vals    = tl.load(feat_ptr + base + hw_off, mask=hw_mask, other=0.0)
    tl.store(out_ptr + base + hw_off, vals * scale.to(vals.dtype), mask=hw_mask)


def _pick_block_hw(HW):
    """Choose BLOCK_HW that divides HW exactly (zero waste), else 4096."""
    if HW % 4096 == 0:
        return 4096
    if HW % 1024 == 0:
        return 1024
    if HW % 512 == 0:
        return 512
    return 4096   # accept ~23% waste for HW=3136, but fewest CTAs


def _fused_lss_impl(in_0, in_1, in_2, in_3):
    """Fused linear + sigmoid + channel-wise feature scaling (C=64, K=8 specialised)."""
    B  = in_2.shape[0]
    HW = in_3.shape[2] * in_3.shape[3]
    out = torch.empty_like(in_3)

    if B == 1:
        # B=1 specialised kernel (b=0 arithmetic eliminated).
        BLOCK_HW = _pick_block_hw(HW)
        grid = (64, triton.cdiv(HW, BLOCK_HW))
        _lss_kernel_b1[grid](in_0, in_1, in_2, in_3, out, HW,
                             BLOCK_HW=BLOCK_HW, num_warps=4)

    else:
        # General kernel with deterministic BLOCK_HW — no autotune noise.
        # BLOCK_HW=4096 for HW≤4096: fewest CTAs, 32 elem/thread (4×128-bit).
        # Use more warps for larger B to improve memory-latency hiding.
        BLOCK_HW = _pick_block_hw(HW)
        NW = 8 if B >= 64 else 4
        grid = (B * 64, triton.cdiv(HW, BLOCK_HW))
        _lss_kernel_fixed[grid](in_0, in_1, in_2, in_3, out, B, HW,
                                BLOCK_HW=BLOCK_HW, num_warps=NW)

    return out


@torch.fx.wrap
def fused_lss_dispatch(in_0, in_1, in_2, in_3, route):
    """Shared dispatch wrapper — identical across all pass files."""
    return _fused_lss_impl(in_0, in_1, in_2, in_3)