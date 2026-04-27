"""
Shared Triton kernels and single dispatch wrapper for all passes.
Importing _dispatch from here guarantees a single unique replacement_func object.
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────
# Embedding + Add + LayerNorm kernel, DIM = 768
# Grid-stride loop: launch NUM_PROGRAMS blocks (≥ 28 SMs × 2)
# so the GPU is well-occupied even for tiny batch×seq lengths.
# LN weight/bias are loaded once per block (not per token).
# ─────────────────────────────────────────────────────────────
@triton.jit
def _kernel_768(
    in_0_ptr, in_4_ptr, in_5_ptr, in_3_ptr, in_2_ptr, in_1_ptr,
    out_ptr,
    B_S,          # total number of tokens (runtime)
    EPS,
    BLOCK_D: tl.constexpr,
):
    DIM = 768
    d    = tl.arange(0, BLOCK_D)
    mask = d < DIM

    # Load LN weight/bias once; reused for every token this block handles
    wt = tl.load(in_2_ptr + d, mask=mask, other=1.0).to(tl.float32)
    bi = tl.load(in_1_ptr + d, mask=mask, other=0.0).to(tl.float32)

    row    = tl.program_id(0)
    stride = tl.num_programs(0)   # = grid size

    while row < B_S:
        word_idx = tl.load(in_0_ptr + row)
        pos_idx  = tl.load(in_5_ptr + row)

        w = tl.load(in_4_ptr + word_idx * DIM + d, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(in_3_ptr + pos_idx  * DIM + d, mask=mask, other=0.0).to(tl.float32)
        x = w + p

        mean     = tl.sum(x, axis=0) / DIM
        centered = x - mean
        c_m      = tl.where(mask, centered, 0.0)
        var      = tl.sum(c_m * c_m, axis=0) / DIM
        rstd     = 1.0 / tl.sqrt(var + EPS)
        x_n      = centered * rstd

        tl.store(out_ptr + row * DIM + d, x_n * wt + bi, mask=mask)
        row += stride


# ─────────────────────────────────────────────────────────────
# Embedding + Add + LayerNorm kernel, DIM = 64 (grid-stride)
# ─────────────────────────────────────────────────────────────
@triton.jit
def _kernel_64(
    in_0_ptr, in_4_ptr, in_5_ptr, in_3_ptr, in_2_ptr, in_1_ptr,
    out_ptr,
    B_S,
    EPS,
    BLOCK_D: tl.constexpr,
):
    DIM = 64
    d   = tl.arange(0, BLOCK_D)

    wt = tl.load(in_2_ptr + d).to(tl.float32)
    bi = tl.load(in_1_ptr + d).to(tl.float32)

    row    = tl.program_id(0)
    stride = tl.num_programs(0)

    while row < B_S:
        word_idx = tl.load(in_0_ptr + row)
        pos_idx  = tl.load(in_5_ptr + row)

        w = tl.load(in_4_ptr + word_idx * DIM + d).to(tl.float32)
        p = tl.load(in_3_ptr + pos_idx  * DIM + d).to(tl.float32)
        x = w + p

        mean     = tl.sum(x, axis=0) / DIM
        centered = x - mean
        var      = tl.sum(centered * centered, axis=0) / DIM
        rstd     = 1.0 / tl.sqrt(var + EPS)
        x_n      = centered * rstd

        tl.store(out_ptr + row * DIM + d, x_n * wt + bi)
        row += stride


# ─────────────────────────────────────────────────────────────
# Position bucket kernel — computes the full N×N matrix.
# N and BLOCK_SIZE are constexprs so Triton specialises per
# unique (N, BLOCK_SIZE) pair (only 3 values in practice).
# ─────────────────────────────────────────────────────────────
@triton.jit
def _kernel_bucket(
    out_ptr,
    N:          tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N * N

    i = (offsets // N).to(tl.int64)
    j = (offsets %  N).to(tl.int64)

    diff     = i - j                                  # tmp_15 = i - j
    offset   = (diff < 0).to(tl.int64) * 16          # 16 where i < j
    abs_diff = tl.abs(diff)

    abs_f   = tl.maximum(abs_diff.to(tl.float32), 0.5)
    log_b   = ((tl.log(abs_f / 8.0) / 2.772588722239781) * 8.0).to(tl.int64)
    clamped = tl.minimum(8 + log_b, 15)

    bucket = tl.where(abs_diff < 8, abs_diff, clamped)
    tl.store(out_ptr + offsets, offset + bucket, mask=mask)


# ─────────────────────────────────────────────────────────────
# Single shared @torch.fx.wrap dispatch wrapper (7 args + route).
# Returns a single tensor matching the single-output pattern.
# ─────────────────────────────────────────────────────────────
@torch.fx.wrap
def _dispatch(in_0, in_4, in_5, in_3, in_2, in_1, route):
    B   = in_0.shape[0]
    S   = in_0.shape[1]
    B_S = B * S
    dev = in_4.device
    dt  = in_4.dtype

    # Use 56 programs (2 per SM on A30) for better occupancy on tiny inputs
    NUM_PROGRAMS = 56

    if route == "route_768_1e5":
        DIM = 768
        out = torch.empty(B, S, DIM, dtype=dt, device=dev)
        _kernel_768[(NUM_PROGRAMS,)](
            in_0, in_4, in_5, in_3, in_2, in_1, out,
            B_S=B_S, EPS=1e-5, BLOCK_D=1024, num_warps=4,
        )
        return out

    elif route == "route_64_1e12":
        DIM = 64
        out = torch.empty(B, S, DIM, dtype=dt, device=dev)
        _kernel_64[(NUM_PROGRAMS,)](
            in_0, in_4, in_5, in_3, in_2, in_1, out,
            B_S=B_S, EPS=1e-12, BLOCK_D=64, num_warps=2,
        )
        return out

    return in_0   # fallback — should never reach