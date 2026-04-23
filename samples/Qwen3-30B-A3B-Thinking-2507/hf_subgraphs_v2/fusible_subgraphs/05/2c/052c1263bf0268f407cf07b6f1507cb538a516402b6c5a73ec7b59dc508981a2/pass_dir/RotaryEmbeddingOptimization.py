import torch
import triton
import triton.language as tl

def pattern(in_2, in_4):
    tmp_1 = in_2[..., :128]
    tmp_2 = in_2[..., 128:]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    return tmp_5

def replacement_args(in_2, in_4):
    return (in_2, in_4)

@triton.jit
def rotary_embed_kernel(
    in2_ptr, in4_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    half_size: tl.constexpr = 128,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in2_val = tl.load(in2_ptr + offsets, mask=mask)
    in4_val = tl.load(in4_ptr + offsets, mask=mask)

    input_index = tl.where(offsets < half_size, half_size + offsets, offsets - half_size)
    in2_val = tl.load(in2_ptr + input_index, mask=mask)
    result = tl.where(offsets < half_size, -in2_val * in4_val, in2_val * in4_val)

    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def rotary_embed(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    rotary_embed_kernel[(num_blocks,)](
        in2_ptr=x,
        in4_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        half_size=128,
    )
    return out

def replacement_func():
    return rotary_embed