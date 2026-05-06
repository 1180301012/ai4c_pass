import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: silu -> mul -> dropout(p=0, training=False)   [identity]
# Input shapes: [1, 257, 1024]  (float32 / bfloat16 / float16)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused SiLU(x) * y
#   out[i] = x[i] * sigmoid(x[i]) * y[i]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_mul_kernel(
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

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sig_x * y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def silu_mul_fused(in_0, in_1):
    """Fused: silu(in_0) * in_1  (dropout p=0 is identity)"""
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    # Grid is a lambda so autotune can vary BLOCK_SIZE
    grid = lambda meta: (
        (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )

    _silu_mul_kernel[grid](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
    )

    return out


def replacement_func():
    return silu_mul_fused