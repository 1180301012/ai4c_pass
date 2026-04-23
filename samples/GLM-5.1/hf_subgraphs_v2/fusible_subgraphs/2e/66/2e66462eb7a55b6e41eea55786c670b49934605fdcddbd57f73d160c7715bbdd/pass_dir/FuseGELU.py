import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
    x_cubed = x * x * x
    inner = 0.044715 * x_cubed + x
    inner = 0.7978845608028654 * inner
    exp_2inner = tl.math.exp(2.0 * inner)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    result = 0.5 * x * (1.0 + tanh_inner)

    tl.store(output_ptr + offsets, result.to(OUTPUT_DTYPE), mask=mask)


@torch.fx.wrap
def gelu_fused(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    output_tensor = torch.empty_like(input_tensor)

    # Map torch dtype to triton dtype
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    output_dtype = dtype_map[input_tensor.dtype]

    gelu_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_DTYPE=output_dtype,
    )

    return (output_tensor,)


def replacement_func():
    return gelu_fused