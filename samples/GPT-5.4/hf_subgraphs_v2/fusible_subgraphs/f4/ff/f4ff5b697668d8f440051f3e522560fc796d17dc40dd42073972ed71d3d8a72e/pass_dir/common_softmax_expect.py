import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def _softmax_only_kernel(
    in_ptr,
    out_ptr,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < row_stride
    ptrs = in_ptr + row_id * row_stride + offs
    vals = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(vals, axis=0)
    num = tl.exp(vals - row_max)
    den = tl.sum(num, axis=0)
    probs = num / den
    tl.store(out_ptr + row_id * row_stride + offs, probs, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def _softmax_expect_kernel(
    in_ptr,
    x_ptr,
    y_ptr,
    softmax_out_ptr,
    coord_out_ptr,
    row_stride,
    coord_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < row_stride

    in_row_ptr = in_ptr + row_id * row_stride + offs
    vals = tl.load(in_row_ptr, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.max(vals, axis=0)
    num = tl.exp(vals - row_max)
    den = tl.sum(num, axis=0)
    probs = num / den

    softmax_row_ptr = softmax_out_ptr + row_id * row_stride + offs
    tl.store(softmax_row_ptr, probs, mask=mask)

    x_vals = tl.load(x_ptr + (offs & 63), mask=mask, other=0).to(tl.float32)
    y_vals = tl.load(y_ptr + (offs >> 6), mask=mask, other=0).to(tl.float32)

    ex = tl.sum(probs * x_vals, axis=0)
    ey = tl.sum(probs * y_vals, axis=0)

    coord_ptr = coord_out_ptr + row_id * coord_row_stride
    tl.store(coord_ptr + 0, ex)
    tl.store(coord_ptr + 1, ey)


@torch.fx.wrap
def softmax_only(in_2):
    out = torch.empty_like(in_2)
    n_rows = in_2.shape[0] * in_2.shape[1]
    row_stride = in_2.shape[2]
    _softmax_only_kernel[(n_rows,)](
        in_ptr=in_2,
        out_ptr=out,
        row_stride=row_stride,
    )
    return out


@torch.fx.wrap
def fused_softmax_spatial_expectation(in_0, in_1, in_2):
    out_softmax = torch.empty_like(in_2)
    out_coord = torch.empty_like(in_2[:, :, :2])

    x_vec = in_0.reshape(64)
    y_vec = in_1.reshape(64)

    n_rows = in_2.shape[0] * in_2.shape[1]
    row_stride = in_2.shape[2]

    _softmax_expect_kernel[(n_rows,)](
        in_ptr=in_2,
        x_ptr=x_vec,
        y_ptr=y_vec,
        softmax_out_ptr=out_softmax,
        coord_out_ptr=out_coord,
        row_stride=row_stride,
        coord_row_stride=2,
    )

    return out_softmax.reshape(-1, 17, 64, 64), out_coord


@torch.fx.wrap
def dispatch_replacement(*args):
    if len(args) == 1:
        return softmax_only(args[0])
    if len(args) == 3:
        return fused_softmax_spatial_expectation(args[0], args[1], args[2])
    route = args[-1]
    if route == "softmax":
        return softmax_only(args[0])
    if route == "full":
        return fused_softmax_spatial_expectation(args[0], args[1], args[2])
    raise RuntimeError(f"Unknown replacement route: {route}")


def replacement_func():
    return dispatch_replacement