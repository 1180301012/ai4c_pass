import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: silu -> element-wise multiply -> dropout(p=0) [no-op]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.ops.aten.silu.default(in_0)
    tmp_1 = torch.ops.aten.mul.Tensor(tmp_0, in_1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused Triton kernel: out[i] = silu(x[i]) * y[i]
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs; padded lanes get 0.0 so they don't affect the result
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    # Triton supports exp for all float dtypes (fp32, fp16, bf16)
    silu_x = x * (1.0 / (1.0 + tl.exp(-x)))

    # Element-wise multiply with the second input
    out = silu_x * y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_mul_dropout(x, y):
    N = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _fused_silu_mul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
    )

    return out


def replacement_func():
    return fused_silu_mul_dropout