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


@triton.jit
def gelu_mul_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)

    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    in_0_f32 = in_0.to(tl.float32)
    gelu_result = in_0_f32 * 0.5 * (1.0 + tl.math.erf(in_0_f32 / sqrt2))
    gelu_result = gelu_result.to(in_0.dtype)

    result = gelu_result * in_1

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gelu_mul_dropout(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_mul_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return gelu_mul_dropout