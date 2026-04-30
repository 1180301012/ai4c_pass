import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return (tmp_1,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def silu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # silu(y) = y * sigmoid(y), then add x
    silu_val = y * tl.sigmoid(y)
    out = silu_val + x

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_add_kernel_autotuned(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # silu(y) = y * sigmoid(y), then add x
    silu_val = y * tl.sigmoid(y)
    out = silu_val + x

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_add(x, y):
    N = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    silu_add_kernel_autotuned[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
    )

    return out


def replacement_func():
    return fused_silu_add