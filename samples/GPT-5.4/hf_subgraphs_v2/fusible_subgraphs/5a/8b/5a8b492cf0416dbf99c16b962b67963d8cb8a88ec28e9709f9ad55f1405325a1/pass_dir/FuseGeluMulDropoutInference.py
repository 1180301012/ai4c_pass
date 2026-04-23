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
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    inv_sqrt2 = 0.70710678118654752440
    gelu_x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    out = gelu_x * y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_gelu_mul_dropout_inference(in_0, in_1):
    out = torch.empty_like(in_0)
    n_elements = out.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    fused_gelu_mul_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_gelu_mul_dropout_inference