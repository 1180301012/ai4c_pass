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
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and upcast to fp32 for numerical stability
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute GELU: result = x / (1 + exp(-inner_doubled))
    # Precomputed doubled inner coefficients to save operations:
    # inner_doubled = 2 * (0.7978845608028654 * x + 0.035676408 * x^3)
    # = 1.5957691216017308 * x + 0.071352816 * x^3
    x_cubed = x * x * x
    inner_doubled = 1.5957691216017308 * x + 0.071352816 * x_cubed
    exp_neg_inner = tl.exp(-inner_doubled)
    result = x / (1.0 + exp_neg_inner)

    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gelu_fused(input_tensor):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    output = torch.empty_like(input_tensor)

    gelu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return gelu_fused