import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "route_384_64")


@triton.jit
def unfold_reshape_kernel_16_8(
    input_ptr,
    output_ptr,
    total_elements,
    L,
    stride_c,
    stride_l,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    k = offset % 9
    remainder = offset // 9
    j = remainder % 8
    i = remainder // 8

    l = i // 2
    g = i % 2

    c = g * 8 + j
    pos = l + k - 4

    valid = (pos >= 0) & (pos < L) & mask

    input_idx = c * stride_c + pos * stride_l

    val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)
    tl.store(output_ptr + offset, val, mask=mask)


@triton.jit
def unfold_reshape_kernel_384_64(
    input_ptr,
    output_ptr,
    total_elements,
    L,
    stride_c,
    stride_l,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    k = offset % 9
    remainder = offset // 9
    j = remainder % 64
    i = remainder // 64

    l = i // 6
    g = i % 6

    c = g * 64 + j
    pos = l + k - 4

    valid = (pos >= 0) & (pos < L) & mask

    input_idx = c * stride_c + pos * stride_l

    val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)
    tl.store(output_ptr + offset, val, mask=mask)


def _run_16_8(in_0):
    L = in_0.shape[2]
    total_elements = L * 16 * 9
    out = torch.empty(L * 2, 8, 9, dtype=in_0.dtype, device=in_0.device)
    stride_c = in_0.stride(1)
    stride_l = in_0.stride(2)
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    unfold_reshape_kernel_16_8[(num_programs,)](
        in_0, out, total_elements, L, stride_c, stride_l,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _run_384_64(in_0):
    L = in_0.shape[2]
    total_elements = L * 384 * 9
    out = torch.empty(L * 6, 64, 9, dtype=in_0.dtype, device=in_0.device)
    stride_c = in_0.stride(1)
    stride_l = in_0.stride(2)
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    unfold_reshape_kernel_384_64[(num_programs,)](
        in_0, out, total_elements, L, stride_c, stride_l,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@torch.fx.wrap
def unfold_reshape_dispatch(in_0, route):
    if route == "route_16_8":
        return _run_16_8(in_0)
    elif route == "route_384_64":
        return _run_384_64(in_0)


def replacement_func():
    return unfold_reshape_dispatch