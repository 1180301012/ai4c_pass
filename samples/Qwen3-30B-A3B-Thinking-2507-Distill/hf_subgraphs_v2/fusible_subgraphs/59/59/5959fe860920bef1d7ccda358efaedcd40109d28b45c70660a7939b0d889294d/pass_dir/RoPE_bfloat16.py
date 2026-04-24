import torch
import triton
import triton.language as tl


@triton.jit
def rope_cos_sin_bf16_kernel(
    x_ptr, cos_ptr, sin_ptr,
    N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row < N:
        d1 = tl.arange(0, BLOCK_D)
        d2 = d1 + D
        m  = d1 < D
        x1 = tl.load(x_ptr + row * D + d1, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + row * D + d2, mask=m, other=0.0).to(tl.float32)
        tl.store(cos_ptr + row * 2 * D + d1, tl.cos(x1).to(tl.bfloat16), mask=m)
        tl.store(cos_ptr + row * 2 * D + d2, tl.cos(x2).to(tl.bfloat16), mask=m)
        tl.store(sin_ptr + row * 2 * D + d1, tl.sin(x1).to(tl.bfloat16), mask=m)
        tl.store(sin_ptr + row * 2 * D + d2, tl.sin(x2).to(tl.bfloat16), mask=m)


@torch.fx.wrap
def rope_bf16(x):
    D = x.shape[-1]
    N = x.numel() // D
    out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.bfloat16, device=x.device)
    return out


@torch.fx.wrap
def rope_bf16_sin(x):
    D = x.shape[-1]
    N = x.numel() // D
    out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.bfloat16, device=x.device)
    return out


def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return rope_bf16


# ── Inserted new kernel (no BLOCK_N, correct autotune key) ──────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def rope_cos_sin_bf16_new(
    x_ptr, cos_ptr, sin_ptr,
    N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row < N:
        d1 = tl.arange(0, BLOCK_D)
        d2 = d1 + D
        m  = d1 < D
        x1 = tl.load(x_ptr + row * D + d1, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + row * D + d2, mask=m, other=0.0).to(tl.float32)
        tl.store(cos_ptr + row * 2 * D + d1, tl.cos(x1).to(tl.bfloat16), mask=m)
        tl.store(cos_ptr + row * 2 * D + d2, tl.cos(x2).to(tl.bfloat16), mask=m)
        tl.store(sin_ptr + row * 2 * D + d1, tl.sin(x1).to(tl.bfloat16), mask=m)
        tl.store(sin_ptr + row * 2 * D + d2, tl.sin(x2).to(tl.bfloat16), mask=m)


@torch.fx.wrap
def rope_bf16_new(x):
    D       = x.shape[-1]
    N       = x.numel() // D
    cos_out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.bfloat16, device=x.device)
    sin_out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.bfloat16, device=x.device)
    rope_cos_sin_bf16_new[(N,)](x, cos_out, sin_out, N, D)
    return cos_out, sin_out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def rope_cos_sin_bf16_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    N,          # total rows  (batch * seq_len)
    D: tl.constexpr,   # head_dim (last dim of input)
    BLOCK_D: tl.constexpr,  # must be >= D and power-of-2
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
            tl.store(cos_ptr + row * 2 * D + d1,     cos1.to(tl.bfloat16), mask=m1)
            tl.store(sin_ptr + row * 2 * D + d1,     sin1.to(tl.bfloat16), mask=m1)

            # Second half [BLOCK_D .. 2*BLOCK_D)
            d2  = d1 + BLOCK_D
            m2  = (d2 < D) & (row < N)
            x2  = tl.load(x_base + d2, mask=m2, other=0.0).to(tl.float32)
            cos2 = tl.cos(x2)
            sin2 = tl.sin(x2)
            tl.store(cos_ptr + row * 2 * D + d2,     cos2.to(tl.bfloat16), mask=m2)
            tl.store(sin_ptr + row * 2 * D + d2,     sin2.to(tl.bfloat16), mask=m2)


@torch.fx.wrap
def rope_bf16(x):
    """Fused RoPE cos/sin kernel – input shape [*, D], outputs [*, 2*D] bfloat16."""
    D      = x.shape[-1]
    N      = x.numel() // D
    cos_out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.bfloat16, device=x.device)
    sin_out = torch.empty((*x.shape[:-1], D * 2), dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_D']),)
    rope_cos_sin_bf16_kernel[(N,)](
        x, cos_out, sin_out,
        N, D,
    )
    return cos_out, sin_out


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement glue  (new clean versions)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    return tmp_3


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return rope_bf16


def pattern_sin(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    return tmp_5


def replacement_args_sin(in_1):
    return (in_1,)


def replacement_func_sin():
    return rope_bf16