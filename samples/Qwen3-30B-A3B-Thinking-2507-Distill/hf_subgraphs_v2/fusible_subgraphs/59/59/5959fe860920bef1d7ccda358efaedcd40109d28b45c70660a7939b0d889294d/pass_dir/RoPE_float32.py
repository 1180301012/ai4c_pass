import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fuse cat + cos + sin + cast to float32
# Input : x  [*, D]  (any shape, last dim = head_dim)
# Output: cos_out [*, 2*D], sin_out [*, 2*D], both float32
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1,  'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 4,  'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 8,  'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 32}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def rope_cos_sin_f32_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    N,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid       = tl.program_id(0)
    row_start = pid * BLOCK_N

    for n_off in tl.static_range(0, BLOCK_N, 1):
        row = row_start + n_off
        if row < N:
            x_base = x_ptr + row * D

            # First half  [0 .. BLOCK_D)
            d1  = tl.arange(0, BLOCK_D)
            m1  = (d1 < D) & (row < N)
            x1  = tl.load(x_base + d1, mask=m1, other=0.0).to(tl.float32)
            cos1 = tl.cos(x1)
            sin1 = tl.sin(x1)
            tl.store(cos_ptr + row * 2 * D + d1, cos1, mask=m1)
            tl.store(sin_ptr + row * 2 * D + d1, sin1, mask=m1)

            # Second half [BLOCK_D .. 2*BLOCK_D)
            d2  = d1 + BLOCK_D
            m2  = (d2 < D) & (row < N)
            x2  = tl.load(x_base + d2, mask=m2, other=0.0).to(tl.float32)
            cos2 = tl.cos(x2)
            sin2 = tl.sin(x2)
            tl.store(cos_ptr + row * 2 * D + d2, cos2, mask=m2)
            tl.store(sin_ptr + row * 2 * D + d2, sin2, mask=m2)


@torch.fx.wrap
def rope_f32(x):
    """Fused RoPE cos/sin kernel – input [*, D], outputs [*, 2*D] float32."""
    D      = x.shape[-1]
    N      = x.numel() // D
    cos_out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.float32, device=x.device)
    sin_out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    rope_cos_sin_f32_kernel[grid](
        x, cos_out, sin_out,
        N, D,
    )
    return cos_out, sin_out


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement glue
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.float32)
    tmp_7 = tmp_5.to(dtype=torch.float32)
    return tmp_6, tmp_7


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return rope_f32