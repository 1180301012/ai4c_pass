import torch
import triton
import triton.language as tl


def pattern(x):
    tmp = x.transpose(-2, -1)
    return tmp


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_H": 32}, num_warps=8, num_stages=2),
    ],
    key=["num_rows", "hidden"],
)
@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    hidden,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_n = offs_n < num_rows
    mask_h = offs_h < hidden
    mask = mask_n[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + pid_b * stride_x0 + offs_n[:, None] * stride_x1 + offs_h[None, :] * stride_x2
    vals = tl.load(x_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + pid_b * stride_o0 + offs_h[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(out_ptrs, tl.trans(vals), mask=mask_h[:, None] & mask_n[None, :])


@torch.fx.wrap
def materialize_transpose_contiguous(x):
    batch = x.shape[0]
    num_rows = x.shape[1]
    hidden = x.shape[2]

    out = torch.empty((batch, hidden, num_rows), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(num_rows, meta["BLOCK_N"]),
        triton.cdiv(hidden, meta["BLOCK_H"]),
        batch,
    )

    _transpose_kernel[grid](
        x,
        out,
        num_rows,
        hidden,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )

    return out


def replacement_func():
    return materialize_transpose_contiguous