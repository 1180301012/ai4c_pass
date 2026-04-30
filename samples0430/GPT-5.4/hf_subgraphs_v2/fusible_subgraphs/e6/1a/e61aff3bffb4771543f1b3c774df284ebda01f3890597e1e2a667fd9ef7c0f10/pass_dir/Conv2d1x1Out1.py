import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "conv")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 256, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 256, "BLOCK_C": 64}, num_warps=8, num_stages=2),
    ],
    key=["S", "C"],
)
@triton.jit
def _conv1x1_out1_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    H,
    W,
    S,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_c,
    out_stride_n,
    out_stride_h,
    out_stride_w,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_h = offs_s // W
    offs_w = offs_s % W

    acc = tl.load(b_ptr).to(tl.float32) + tl.zeros([BLOCK_S], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        x = tl.load(
            x_ptr
            + pid_n * x_stride_n
            + offs_c[:, None] * x_stride_c
            + offs_h[None, :] * x_stride_h
            + offs_w[None, :] * x_stride_w,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + offs_c * w_stride_c, mask=mask_c, other=0.0).to(tl.float32)
        acc += tl.sum(x * w[:, None], axis=0)

    tl.store(
        out_ptr + pid_n * out_stride_n + offs_h * out_stride_h + offs_w * out_stride_w,
        acc,
        mask=mask_s,
    )


@triton.jit
def _softmax_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    dim1,
    cols,
    x_stride0,
    x_stride1,
    x_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    idx0 = row // dim1
    idx1 = row % dim1

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < cols

    x_row_ptr = x_ptr + idx0 * x_stride0 + idx1 * x_stride1
    out_row_ptr = out_ptr + idx0 * out_stride0 + idx1 * out_stride1

    x = tl.load(x_row_ptr + offs * x_stride2, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(out_row_ptr + offs * out_stride2, y, mask=mask)


@torch.fx.wrap
def _dispatch(*args):
    route = args[-1]

    if route == "conv":
        bias, weight, x, _ = args
        n, c, h, w = x.shape
        s = h * w
        out = torch.empty((n, 1, h, w), device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(s, META["BLOCK_S"]), n)
        _conv1x1_out1_kernel[grid](
            x,
            weight,
            bias,
            out,
            c,
            h,
            w,
            s,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            weight.stride(1),
            out.stride(0),
            out.stride(2),
            out.stride(3),
        )
        return out

    if route == "softmax":
        x, _ = args
        dim0, dim1, cols = x.shape
        out = torch.empty((dim0, dim1, cols, 1), device=x.device, dtype=x.dtype)
        block_size = 1
        while block_size < cols:
            block_size *= 2
        num_warps = 4
        if block_size >= 1024:
            num_warps = 8
        rows = dim0 * dim1
        _softmax_unsqueeze_kernel[(rows,)](
            x,
            out,
            dim1,
            cols,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return out

    return args[0]


def replacement_func():
    return _dispatch