import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((128, 128))
    return (tmp_3, tmp_4, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_128_128")


@triton.jit
def mul_expand_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_expand_ptr, out_mul_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * D
    mask = offsets < n_elements

    row = offsets // D

    # Broadcast multiply: in_1[row] * in_2[row, col]
    weight = tl.load(in_1_ptr + row, mask=mask, other=0.0)
    x = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    result = weight * x
    tl.store(out_mul_ptr + offsets, result, mask=mask)

    # Expand in_0[row] to [N, D]
    idx = tl.load(in_0_ptr + row, mask=mask, other=0)
    tl.store(out_expand_ptr + offsets, idx, mask=mask)


@triton.jit
def zeros_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    zero_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    tl.store(out_ptr + offsets, zero_vals, mask=mask)


def _run_1000_16(in_0, in_1, in_2):
    N = in_1.shape[0]
    D = in_2.shape[1]
    n_elements = N * D
    BLOCK_SIZE = 1024

    out_mul = torch.empty((N, D), dtype=in_2.dtype, device=in_2.device)
    out_expand = torch.empty((N, D), dtype=in_0.dtype, device=in_0.device)
    out_zeros = torch.empty((1000, 16), dtype=in_2.dtype, device=in_2.device)

    grid1 = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mul_expand_kernel[grid1](
        in_0, in_1, in_2,
        out_expand, out_mul,
        N, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    zeros_size = 1000 * 16
    grid2 = ((zeros_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    zeros_kernel[grid2](out_zeros, zeros_size, BLOCK_SIZE=BLOCK_SIZE)

    return (out_expand, out_zeros, out_mul)


def _run_128_128(in_0, in_1, in_2):
    N = in_1.shape[0]
    D = in_2.shape[1]
    n_elements = N * D
    BLOCK_SIZE = 1024

    out_mul = torch.empty((N, D), dtype=in_2.dtype, device=in_2.device)
    out_expand = torch.empty((N, D), dtype=in_0.dtype, device=in_0.device)
    out_zeros = torch.empty((128, 128), dtype=in_2.dtype, device=in_2.device)

    grid1 = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mul_expand_kernel[grid1](
        in_0, in_1, in_2,
        out_expand, out_mul,
        N, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    zeros_size = 128 * 128
    grid2 = ((zeros_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    zeros_kernel[grid2](out_zeros, zeros_size, BLOCK_SIZE=BLOCK_SIZE)

    return (out_expand, out_zeros, out_mul)


@torch.fx.wrap
def fused_dispatch(in_0, in_1, in_2, route):
    if route == "route_1000_16":
        return _run_1000_16(in_0, in_1, in_2)
    else:
        return _run_128_128(in_0, in_1, in_2)


def replacement_func():
    return fused_dispatch