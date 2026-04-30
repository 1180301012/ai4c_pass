import torch
import triton
import triton.language as tl


@triton.jit
def _bool_cast_kernel(
    in_ptr,
    out_bool_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out_bool = x != 0
    tl.store(out_bool_ptr + offsets, out_bool, mask=mask)


@torch.fx.wrap
def triton_bool_cast(in_0):
    n_elements = in_0.numel()
    out_bool = torch.empty(in_0.shape, device=in_0.device, dtype=torch.bool)

    block_size = 1024
    grid = ((n_elements + block_size - 1) // block_size,)

    _bool_cast_kernel[grid](
        in_0,
        out_bool,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out_bool


def shared_replacement_func():
    return triton_bool_cast