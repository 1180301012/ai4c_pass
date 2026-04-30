import torch
import triton
import triton.language as tl


_INV_SCALE = 0.08838834764831843


def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_div_relu_square_contiguous_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x * _INV_SCALE, 0.0)
    out = y * y

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_div_relu_square_strided_3d_kernel(
    x_ptr,
    out_ptr,
    dim0,
    dim1,
    dim2,
    x_stride0,
    x_stride1,
    x_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    d2 = offsets % dim2
    tmp = offsets // dim2
    d1 = tmp % dim1
    d0 = tmp // dim1

    x_ptrs = x_ptr + d0 * x_stride0 + d1 * x_stride1 + d2 * x_stride2
    out_ptrs = out_ptr + d0 * out_stride0 + d1 * out_stride1 + d2 * out_stride2

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x * _INV_SCALE, 0.0)
    out = y * y

    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def triton_fused_div_relu_square(in_0):
    out = torch.empty_like(in_0)
    n_elements = in_0.numel()
    if n_elements == 0:
        return out

    if n_elements <= 512:
        block_size = 256
        num_warps = 1
    elif n_elements <= 4096:
        block_size = 512
        num_warps = 2
    else:
        block_size = 1024
        num_warps = 4

    grid = (triton.cdiv(n_elements, block_size),)

    if in_0.is_contiguous() and out.is_contiguous():
        fused_div_relu_square_contiguous_kernel[grid](
            in_0,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return out

    fused_div_relu_square_strided_3d_kernel[grid](
        in_0,
        out,
        in_0.shape[0],
        in_0.shape[1],
        in_0.shape[2],
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


def replacement_func():
    return triton_fused_div_relu_square