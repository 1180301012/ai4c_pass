import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _relu_copy_contiguous_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _relu_inplace_contiguous_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(x_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_copy_strided_4d_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    inner_size,
    dim1,
    dim2,
    dim3,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    n = offsets // inner_size
    inner = offsets % inner_size
    dim23 = dim2 * dim3
    c = inner // dim23
    rem = inner % dim23
    h = rem // dim3
    w = rem % dim3

    x_offsets = n * x_stride0 + c * x_stride1 + h * x_stride2 + w * x_stride3
    out_offsets = n * out_stride0 + c * out_stride1 + h * out_stride2 + w * out_stride3

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(out_ptr + out_offsets, y, mask=mask)


@triton.jit
def _relu_inplace_strided_4d_kernel(
    x_ptr,
    n_elements,
    inner_size,
    dim1,
    dim2,
    dim3,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    n = offsets // inner_size
    inner = offsets % inner_size
    dim23 = dim2 * dim3
    c = inner // dim23
    rem = inner % dim23
    h = rem // dim3
    w = rem % dim3

    x_offsets = n * x_stride0 + c * x_stride1 + h * x_stride2 + w * x_stride3

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(x_ptr + x_offsets, y, mask=mask)


def _is_default_contiguous_4d(shape, strides):
    if len(shape) != 4:
        return False
    return (
        strides[3] == 1
        and strides[2] == shape[3]
        and strides[1] == shape[2] * shape[3]
        and strides[0] == shape[1] * shape[2] * shape[3]
    )


@torch.fx.wrap
def relu_dispatch(x, route):
    shape = x.shape
    total = x.numel()
    strides = x.stride()

    if route == "inplace":
        if _is_default_contiguous_4d(shape, strides):
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            _relu_inplace_contiguous_kernel[grid](x, total)
        else:
            inner = total // shape[0]
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            _relu_inplace_strided_4d_kernel[grid](
                x,
                total,
                inner,
                shape[1],
                shape[2],
                shape[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                BLOCK_SIZE=256,
                num_warps=2,
            )
        return x

    if route == "out":
        out = torch.empty_like(x)
        out_strides = out.stride()
        if _is_default_contiguous_4d(shape, strides) and _is_default_contiguous_4d(shape, out_strides):
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            _relu_copy_contiguous_kernel[grid](x, out, total)
        else:
            inner = total // shape[0]
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            _relu_copy_strided_4d_kernel[grid](
                x,
                out,
                total,
                inner,
                shape[1],
                shape[2],
                shape[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                out_strides[0],
                out_strides[1],
                out_strides[2],
                out_strides[3],
                BLOCK_SIZE=256,
                num_warps=2,
            )
        return out

    raise RuntimeError(f"Unknown route: {route}")