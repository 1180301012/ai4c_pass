import torch
import triton
import triton.language as tl


# Match only the tail that is profitable and safe to replace under the sandbox.
def pattern(x, residual):
    tmp_3 = torch.nn.functional.dropout(x, 0.0, False, False)
    tmp_4 = tmp_3 + residual
    return tmp_4


def replacement_args(x, residual):
    return (x, residual)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    out = x + residual
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_dropout_add(x, residual):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    add_kernel[grid](x, residual, out, n_elements)
    return out


def replacement_func():
    return fused_dropout_add