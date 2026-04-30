import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_3 = x.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=2),
    ],
    key=["num_rows"],
)
@triton.jit
def _transpose_gelu_kernel_512(
    x_ptr,
    out_ptr,
    num_rows,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, 512)
    mask_n = offs_n < num_rows

    x_ptrs = x_ptr + pid_b * stride_x0 + offs_n[:, None] * stride_x1 + offs_h[None, :] * stride_x2
    x = tl.load(x_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    gelu = x * (0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475)))

    out_ptrs = out_ptr + pid_b * stride_o0 + offs_h[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(out_ptrs, tl.trans(gelu), mask=mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_H": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_H": 64}, num_warps=8, num_stages=2),
    ],
    key=["num_rows", "hidden"],
)
@triton.jit
def _transpose_gelu_kernel_generic(
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
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    gelu = x * (0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475)))

    out_ptrs = out_ptr + pid_b * stride_o0 + offs_h[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(out_ptrs, tl.trans(gelu), mask=mask_h[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_transpose_gelu(x):
    batch = x.shape[0]
    num_rows = x.shape[1]
    hidden = x.shape[2]

    out = torch.empty((batch, hidden, num_rows), device=x.device, dtype=x.dtype)

    if hidden == 512:
        grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_N"]), batch)
        _transpose_gelu_kernel_512[grid](
            x,
            out,
            num_rows,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )
    else:
        grid = lambda meta: (
            triton.cdiv(num_rows, meta["BLOCK_N"]),
            triton.cdiv(hidden, meta["BLOCK_H"]),
            batch,
        )
        _transpose_gelu_kernel_generic[grid](
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
    return fused_transpose_gelu