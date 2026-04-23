import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_3 = x.transpose(-2, -1)
    return tmp_3


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=8),
    ],
    key=["n_time", "n_cols"],
)
@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    n_time,
    n_cols,
    stride_xb,
    stride_xt,
    stride_xc,
    stride_ob,
    stride_oc,
    stride_ot,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < n_time) & (offs_n[None, :] < n_cols)

    x_offsets = (
        pid_b * stride_xb
        + offs_m[:, None] * stride_xt
        + offs_n[None, :] * stride_xc
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0)

    out_offsets = (
        pid_b * stride_ob
        + offs_n[:, None] * stride_oc
        + offs_m[None, :] * stride_ot
    )
    tl.store(out_ptr + out_offsets, tl.trans(x), mask=tl.trans(mask))


@torch.fx.wrap
def materialize_transpose_2d(x):
    n_batch = x.shape[0]
    n_time = x.shape[-2]
    n_cols = x.shape[-1]

    out = torch.empty((n_batch, n_cols, n_time), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(n_time, meta["BLOCK_M"]),
        triton.cdiv(n_cols, meta["BLOCK_N"]),
        n_batch,
    )

    _transpose_kernel[grid](
        x,
        out,
        n_batch,
        n_time,
        n_cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )

    return out


def replacement_func():
    return materialize_transpose_2d