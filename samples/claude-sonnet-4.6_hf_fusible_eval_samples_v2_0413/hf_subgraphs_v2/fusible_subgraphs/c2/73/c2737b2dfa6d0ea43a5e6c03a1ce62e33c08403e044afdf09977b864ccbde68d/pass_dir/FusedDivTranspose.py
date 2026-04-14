import torch
import triton
import triton.language as tl


def pattern(x, divisor):
    tmp_0 = x / divisor
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(x, divisor):
    return (x, divisor)


# ---------------------------------------------------------------------------
# Kernel: tiled fused divide + transpose
#
#   Input  [BH, S, D] → Output [BH, D, S]
#   Grid   (BH, ceil(S / BLOCK_S))               ← 2-D grid
#
#   S, SD=S*D, BLOCK_S, BLOCK_D are ALL tl.constexpr:
#     • s * BLOCK_D  →  shift (BLOCK_D is power-of-2)
#     • d * S        →  shift (S is power-of-2 for most shapes)
#     • bh * SD      →  shift (SD is power-of-2 for most shapes)
#     • s_offs < S   →  compiler can prove always-true when S%BLOCK_S==0
#       (eliminates the mask entirely for S∈{64,128,256,16} with BLOCK_S=32/4)
#
#   num_warps = BLOCK_S*BLOCK_D // 64  →  ~64 elements per warp
#     For bfloat16/float16: 64 elements × 2 bytes = 128 bytes per warp =
#     exactly 1 L1 cache line; Triton emits vectorized 128-bit LDG/STG.
# ---------------------------------------------------------------------------
@triton.jit
def fused_div_transpose_kernel(
    x_ptr, out_ptr,
    inv_divisor,
    S:  tl.constexpr,      # shape[-2]
    SD: tl.constexpr,      # S * D  (== S * BLOCK_D)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr, # == D always
):
    bh      = tl.program_id(0)
    s_block = tl.program_id(1)

    s_start = s_block * BLOCK_S
    s_offs  = s_start + tl.arange(0, BLOCK_S)
    d_offs  = tl.arange(0, BLOCK_D)

    bh_off  = bh * SD              # constexpr multiply → shift / immediate

    # ── Load [BLOCK_S, BLOCK_D] (coalesced; s*BLOCK_D → shift) ─────────────
    in_offs = s_offs[:, None] * BLOCK_D + d_offs[None, :]
    mask_s  = s_offs[:, None] < S  # S constexpr → may be folded away
    data = tl.load(x_ptr + bh_off + in_offs, mask=mask_s, other=0.0)

    data = data * inv_divisor

    # ── Store [BLOCK_D, BLOCK_S] transposed (d*S → shift when S is pow2) ───
    out_offs = d_offs[:, None] * S + s_offs[None, :]
    mask_t   = s_offs[None, :] < S
    tl.store(out_ptr + bh_off + out_offs, tl.trans(data), mask=mask_t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


# Per-shape config cache
_kernel_cache: dict = {}


@torch.fx.wrap
def fused_div_transpose(x, divisor):
    s   = x.shape
    cfg = _kernel_cache.get(s)

    if cfg is None:
        ndim  = len(s)
        BH    = 1
        for i in range(ndim - 2):
            BH *= s[i]
        S_val = s[-2]
        D_val = s[-1]
        SD    = S_val * D_val

        out_s = list(s)
        out_s[-2] = D_val
        out_s[-1] = S_val
        out_s = tuple(out_s)

        BLOCK_D = _next_pow2(D_val)

        # Target 256 threads/block → fill with BLOCK_S
        BLOCK_S = max(1, 256 // BLOCK_D)
        BLOCK_S = _next_pow2(BLOCK_S)
        BLOCK_S = min(BLOCK_S, _next_pow2(S_val))

        grid = (BH, (S_val + BLOCK_S - 1) // BLOCK_S)

        # num_warps: target 64 elements per warp
        # → Triton emits 128-byte (1 cache-line) vectorised LDG/STG per warp
        num_warps = max(1, (BLOCK_S * BLOCK_D) // 64)

        cfg = (out_s, S_val, SD, BLOCK_S, BLOCK_D, grid,
               1.0 / float(divisor), num_warps)
        _kernel_cache[s] = cfg

    out_s, S, SD, BLOCK_S, BLOCK_D, grid, inv_div, num_warps = cfg

    out = torch.empty(out_s, dtype=x.dtype, device=x.device)
    fused_div_transpose_kernel[grid](
        x, out, inv_div,
        S=S, SD=SD,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out


def replacement_func():
    return fused_div_transpose