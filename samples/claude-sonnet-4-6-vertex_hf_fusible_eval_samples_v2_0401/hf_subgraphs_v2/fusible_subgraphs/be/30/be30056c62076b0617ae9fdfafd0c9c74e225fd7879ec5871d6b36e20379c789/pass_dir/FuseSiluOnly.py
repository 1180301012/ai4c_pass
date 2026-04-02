import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fallback: match only the SiLU activation so we at least get some benefit
# if the fuller split-pattern doesn't match.
# ---------------------------------------------------------------------------
def pattern(in_1):
    return torch.nn.functional.silu(in_1, inplace=True)


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _silu_kernel(
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
    y = (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def silu_triton(in_1):
    N = in_1.numel()
    out = torch.empty_like(in_1)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _silu_kernel[grid](in_1, out, N)
    return out


def replacement_func():
    return silu_triton