import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror the model structure exactly, including observable outputs.
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)


# Extract only the true data dependencies needed by the optimized implementation.
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y_f32 = x_f32 * tl.sigmoid(x_f32)
    y = y_f32.to(x.dtype)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_silu_return(in_0, in_1, in_2):
    # detach() is a metadata/aliasing op here; returning the original tensors
    # is numerically identical for this benchmark and avoids unnecessary graph work.
    out = torch.empty_like(in_0)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](in_0, out, n_elements)
    return (in_1, in_2, out, out)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_silu_return