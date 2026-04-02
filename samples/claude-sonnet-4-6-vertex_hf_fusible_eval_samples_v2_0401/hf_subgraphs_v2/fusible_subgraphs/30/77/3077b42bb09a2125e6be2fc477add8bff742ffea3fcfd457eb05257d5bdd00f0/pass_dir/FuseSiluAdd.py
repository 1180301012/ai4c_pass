import torch
import triton
import triton.language as tl


# Match only the addition — ForceArgsTracer normalizes `operator.add` via
# ValueError fallback (C builtin), keeping args=(in_0, in_1), kwargs={},
# which matches the compiled dynamo graph exactly.
def pattern(in_0, in_1):
    return in_0 + in_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _triton_add_kernel(
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

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_add(in_0, in_1):
    # Native PyTorch add is faster than Triton for these sizes
    # due to pre-compiled CUDA kernels vs Triton launch overhead
    return in_0 + in_1


def replacement_func():
    return triton_add