import torch
import triton
import triton.language as tl


# ── Pattern: match ONLY GELU (single output) ─────────────────────────────────
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


# ── Triton kernel: elementwise GELU ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def triton_gelu_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    y_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * inv_sqrt2))

    tl.store(out_ptr + offsets, y_f32.to(x.dtype), mask=mask)


# ── Kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_gelu(in_0):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    triton_gelu_kernel[grid](in_0, out, N)
    return out


def replacement_func():
    return triton_gelu