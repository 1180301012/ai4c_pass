import torch
import triton
import triton.language as tl


def pattern(mask_input):
    mask_float = mask_input.to(dtype=torch.float32)
    sub = 1.0 - mask_float
    result = sub * -3.4028234663852886e+38
    return result


def replacement_args(mask_input):
    return (mask_input,)


@triton.jit
def attention_mask_kernel(
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load int64 values and convert to float32
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Compute (1 - mask_val) * -inf
    result = (1.0 - mask_val) * -3.4028234663852886e+38

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask(mask_input):
    n_elements = mask_input.numel()
    output = torch.empty_like(mask_input, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    attention_mask_kernel[grid](
        mask_input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_attention_mask