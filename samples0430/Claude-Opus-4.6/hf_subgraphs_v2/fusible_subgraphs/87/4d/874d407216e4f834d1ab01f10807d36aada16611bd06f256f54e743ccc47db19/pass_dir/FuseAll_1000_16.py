import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def broadcast_mul_kernel(
    in_1_ptr, in_2_ptr, out_ptr,
    N, D, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // D

    # Broadcast multiply: in_1[row] * in_2[row, col]
    weight = tl.load(in_1_ptr + row, mask=mask, other=0.0)
    x = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    result = weight * x
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def broadcast_mul(in_1, in_2):
    N = in_1.shape[0]
    D = in_2.shape[1]
    n_elements = N * D

    out = torch.empty((N, D), dtype=in_2.dtype, device=in_2.device)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    broadcast_mul_kernel[grid](
        in_1, in_2, out,
        N, D, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return broadcast_mul