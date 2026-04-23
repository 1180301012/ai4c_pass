import torch
import triton
import triton.language as tl


def pattern(conv2d_result, in_3):
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d_result
    return tmp_4


def replacement_args(conv2d_result, in_3):
    return (conv2d_result, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_hardtanh_mul_kernel(
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # hardtanh(x, 0.0, 6.0) = clamp(x, 0, 6)
    x_clamped = tl.minimum(tl.maximum(x, 0.0), 6.0)

    out = x_clamped * y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_hardtanh_mul(x, y):
    n_elements = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_hardtanh_mul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
    )

    return out


def replacement_func():
    return fused_hardtanh_mul