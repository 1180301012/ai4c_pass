import torch
import triton
import triton.language as tl


# Test: match just scale (0.0625 * x) to find what FX node it creates
def pattern(x):
    return 0.0625 * x


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=16),
        triton.Config({'BLOCK': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def scale_kernel(
    in_ptr, out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 0.0625, mask=mask)


@torch.fx.wrap
def fused_scale_softmax_matmul_permute(x):
    out = torch.empty_like(x)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta['BLOCK']),)

    scale_kernel[grid](x, out, n)
    return out


def replacement_func():
    return fused_scale_softmax_matmul_permute