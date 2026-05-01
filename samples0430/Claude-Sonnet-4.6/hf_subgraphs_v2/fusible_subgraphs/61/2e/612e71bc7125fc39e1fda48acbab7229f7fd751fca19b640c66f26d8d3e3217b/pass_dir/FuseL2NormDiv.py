import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = x.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = x / tmp_0
    return tmp_1


def replacement_args(x):
    return (x,)


# ── Kernel A: D == BLOCK_SIZE (power-of-2 D, zero masking overhead) ─────────
@triton.jit
def l2_normalize_exact_kernel(
    x_ptr, out_ptr,
    D: tl.constexpr,          # compile-time size → unmasked load/store
):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_idx * D + offs).to(tl.float32)
    inv_norm = tl.rsqrt(tl.sum(x * x, axis=0))
    tl.store(out_ptr + row_idx * D + offs, (x * inv_norm).to(tl.bfloat16))


# ── Kernel B: D != power-of-2 (masked, but mask is compile-time bitmap) ─────
@triton.jit
def l2_normalize_padded_kernel(
    x_ptr, out_ptr,
    D: tl.constexpr,          # constexpr → mask folded at compile time
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D            # compile-time constant bitmap
    x = tl.load(x_ptr + row_idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    inv_norm = tl.rsqrt(tl.sum(x * x, axis=0))
    tl.store(out_ptr + row_idx * D + offs, (x * inv_norm).to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def fused_l2_normalize(x):
    N, D = x.shape[0], x.shape[1]
    BLOCK_SIZE = 1 << (D - 1).bit_length()   # next pow-2 >= D
    out = torch.empty_like(x)
    if D == BLOCK_SIZE:
        # D is already a power-of-2: no masking at all
        l2_normalize_exact_kernel[(N,)](x, out, D=D, num_warps=1)
    else:
        # D is not a power-of-2: compile-time mask covers the tail
        l2_normalize_padded_kernel[(N,)](x, out, D=D, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
    return out


def replacement_func():
    return fused_l2_normalize