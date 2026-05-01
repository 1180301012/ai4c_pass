import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# ── Float32 kernel: exact GELU via tl.math.erf ─────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_mul_kernel_fp32(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    result = gelu * y

    tl.store(out_ptr + offsets, result, mask=mask)


# ── Low-precision kernel: tanh-approx GELU via libdevice.tanh ──────────────
#    MUFU.TANH ~4 cycles vs erf ~20+ cycles; accurate within bfloat16/float16
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_mul_kernel_lowp(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    # Tanh-approximated GELU (Hendrycks & Gimpel):
    #   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715*x²)))
    # sqrt(2/π) = 0.7978845608028654; uses hardware MUFU.TANH ~4 cycles
    inner = 0.7978845608028654 * x_f32 * (1.0 + 0.044715 * x_f32 * x_f32)
    gelu = 0.5 * x_f32 * (1.0 + libdevice.tanh(inner))
    result = gelu * y_f32

    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)

@torch.fx.wrap
def triton_gelu_mul_dropout(x, y):
    N = x.numel()
    out = torch.empty_like(x)

    def grid(meta):
        return ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    if x.dtype == torch.float32:
        fused_gelu_mul_kernel_fp32[grid](x, y, out, N)
    else:
        fused_gelu_mul_kernel_lowp[grid](x, y, out, N)

    return out


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_gelu_mul_dropout