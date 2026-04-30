import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_gelu_mul_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs (Triton infers native dtype from pointer)
    x_raw = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    y     = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Upcast to fp32 for accurate GELU computation
    x = x_raw.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    cdf = 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # Multiply in fp32, auto-cast back to pointer dtype on store
    result = x * cdf * y
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_gelu_mul(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_gelu_mul_kernel[grid](in_0, in_1, out, N)
    return out


def replacement_func():
    return fused_gelu_mul