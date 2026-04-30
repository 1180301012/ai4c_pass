import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0: torch.Tensor):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


# Argument extraction function
def replacement_args(in_0: torch.Tensor):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def relu_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.maximum(x, 0)
    tl.store(x_ptr + offsets, x, mask=mask)


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def relu_inplace_flatten(in_0: torch.Tensor):
    # Safe fallback for unexpected non-CUDA / non-contiguous inputs.
    if (not in_0.is_cuda) or (not in_0.is_contiguous()):
        in_0.relu_()
        return in_0.flatten(1, -1)

    n_elements = in_0.numel()
    if n_elements != 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        relu_inplace_kernel[grid](
            in_0,
            n_elements,
        )
    return in_0.flatten(1, -1)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return relu_inplace_flatten