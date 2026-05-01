import torch
import triton
import triton.language as tl


def pattern(x, y):
    s = torch.sigmoid(x)
    r = y * s
    return r


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    DTYPE_IS_FP16: tl.constexpr,
    DTYPE_IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = y * sigmoid(x)"""
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    result = y * tl.sigmoid(x)

    if DTYPE_IS_FP16:
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
    elif DTYPE_IS_BF16:
        tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_sigmoid_mul(x, y):
    out       = torch.empty_like(y)
    n_elements = x.numel()
    dtype     = y.dtype
    is_fp16   = (dtype == torch.float16)
    is_bf16   = (dtype == torch.bfloat16)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_mul_kernel[grid](
        x, y, out,
        n_elements,
        DTYPE_IS_FP16=is_fp16,
        DTYPE_IS_BF16=is_bf16,
    )
    return out


def replacement_func():
    return triton_sigmoid_mul