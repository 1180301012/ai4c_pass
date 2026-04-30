import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_gelu_pad_kernel(
    input_ptr,
    output_ptr,
    n_input_elements,
    n_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_output_elements

    # Load input values only for valid indices
    input_mask = offsets < n_input_elements
    input_vals = tl.load(input_ptr + offsets, mask=input_mask, other=0.0).to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    gelu_val = input_vals * 0.5 * (1.0 + tl.libdevice.erf(input_vals / sqrt2))

    # For input region, output is GELU; for padding region, output is 0
    output_vals = tl.where(input_mask, gelu_val, 0.0)

    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def fused_gelu_reshape_pad(in_0):
    n_input = in_0.numel()  # 190464 for [1, 124, 1536]
    n_output = 249 * 768  # 190848 for [1, 249, 768]

    output = torch.empty((1, 249, 768), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 1024
    grid = ((n_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_gelu_pad_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        n_input_elements=n_input,
        n_output_elements=n_output,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

def replacement_func():
    return fused_gelu_reshape_pad