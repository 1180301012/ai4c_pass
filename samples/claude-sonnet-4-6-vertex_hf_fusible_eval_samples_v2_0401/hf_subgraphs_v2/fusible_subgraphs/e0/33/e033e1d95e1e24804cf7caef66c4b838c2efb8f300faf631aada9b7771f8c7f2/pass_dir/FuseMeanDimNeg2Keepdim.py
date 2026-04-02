import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: x.mean(dim=-2, keepdim=True) on a 3-D tensor [B, N, C]
# ---------------------------------------------------------------------------

def pattern(in_2):
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4


def replacement_args(in_2):
    return (in_2,)


# ---------------------------------------------------------------------------
# Triton kernel – single-pass reduction
#
# Grid: (B, cdiv(C, BLOCK_C))
# Each program accumulates [BLOCK_N, BLOCK_C] tiles over the N dimension.
# Inner-C dimension is contiguous → coalesced loads within a row.
# Accumulation in fp32; IS_FP16/IS_BF16 constexprs cast in-kernel so no
# extra host-side .to() kernel launch is needed.
# B is included in the autotune key so each batch size gets its own best
# (BLOCK_N, BLOCK_C) pair.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ── Small B (B≤4): tiny grid → maximise programs with small BLOCK_C ──
        triton.Config({'BLOCK_N':  64, 'BLOCK_C':  8}, num_warps=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C':  8}, num_warps=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C':  8}, num_warps=4),
        triton.Config({'BLOCK_N':  64, 'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 16}, num_warps=4),
    ],
    key=['B', 'N', 'C'],
)
@triton.jit
def _mean_neg2_small(
    input_ptr, output_ptr,
    B, N, C,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
):
    b     = tl.program_id(0)
    c_pid = tl.program_id(1)
    c_offs = c_pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        ptrs   = input_ptr + b * N * C + n_offs[:, None] * C + c_offs[None, :]
        mask2d = n_mask[:, None] & c_mask[None, :]
        acc   += tl.sum(tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32), axis=0)
    mean_f32 = acc / N
    if IS_FP16:
        out_vals = mean_f32.to(tl.float16)
    elif IS_BF16:
        out_vals = mean_f32.to(tl.bfloat16)
    else:
        out_vals = mean_f32
    tl.store(output_ptr + b * C + c_offs, out_vals, mask=c_mask)


@triton.autotune(
    configs=[
        # ── Medium B (5≤B≤24): BLOCK_C=32/64 + pipeline for ~96-128 programs ──
        triton.Config({'BLOCK_N':  64, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N':  64, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
    ],
    key=['B', 'N', 'C'],
)
@triton.jit
def _mean_neg2_medium(
    input_ptr, output_ptr,
    B, N, C,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
):
    b     = tl.program_id(0)
    c_pid = tl.program_id(1)
    c_offs = c_pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        ptrs   = input_ptr + b * N * C + n_offs[:, None] * C + c_offs[None, :]
        mask2d = n_mask[:, None] & c_mask[None, :]
        acc   += tl.sum(tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32), axis=0)
    mean_f32 = acc / N
    if IS_FP16:
        out_vals = mean_f32.to(tl.float16)
    elif IS_BF16:
        out_vals = mean_f32.to(tl.bfloat16)
    else:
        out_vals = mean_f32
    tl.store(output_ptr + b * C + c_offs, out_vals, mask=c_mask)


@triton.autotune(
    configs=[
        # ── Large B (B>24): BLOCK_C=128/256 maximises coalescing + pipeline ──
        triton.Config({'BLOCK_N':  64, 'BLOCK_C': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_N':  64, 'BLOCK_C': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_N':  64, 'BLOCK_C': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 256}, num_warps=16, num_stages=2),
    ],
    key=['B', 'N', 'C'],
)
@triton.jit
def _mean_neg2_large(
    input_ptr, output_ptr,
    B, N, C,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
):
    b     = tl.program_id(0)
    c_pid = tl.program_id(1)
    c_offs = c_pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        ptrs   = input_ptr + b * N * C + n_offs[:, None] * C + c_offs[None, :]
        mask2d = n_mask[:, None] & c_mask[None, :]
        acc   += tl.sum(tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32), axis=0)
    mean_f32 = acc / N
    if IS_FP16:
        out_vals = mean_f32.to(tl.float16)
    elif IS_BF16:
        out_vals = mean_f32.to(tl.bfloat16)
    else:
        out_vals = mean_f32
    tl.store(output_ptr + b * C + c_offs, out_vals, mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# Dispatches to the right specialised kernel based on batch size.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_mean_dim_neg2_keepdim(x):
    B = x.shape[0]
    N = x.shape[1]
    C = x.shape[2]

    # Output allocated directly in input dtype – no separate .to() needed.
    out = torch.empty((B, 1, C), dtype=x.dtype, device=x.device)

    is_fp16 = (x.dtype == torch.float16)
    is_bf16 = (x.dtype == torch.bfloat16)
    grid    = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))

    if B <= 4:
        _mean_neg2_small[grid](x, out, B, N, C, IS_FP16=is_fp16, IS_BF16=is_bf16)
    elif B <= 24:
        _mean_neg2_medium[grid](x, out, B, N, C, IS_FP16=is_fp16, IS_BF16=is_bf16)
    else:
        _mean_neg2_large[grid](x, out, B, N, C, IS_FP16=is_fp16, IS_BF16=is_bf16)

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_mean_dim_neg2_keepdim