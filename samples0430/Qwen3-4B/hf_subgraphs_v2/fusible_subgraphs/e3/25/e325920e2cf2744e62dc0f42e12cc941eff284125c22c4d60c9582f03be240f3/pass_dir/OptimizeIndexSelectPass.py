import torch
import triton
import triton.language as tl

def pattern(a, b):
    return (a[1], b.index_select(-2, a[0]))

def replacement_args(a, b):
    return (a, b)

@triton.jit
def index_select_kernel(
    indices_ptr,
    b_ptr,
    output_ptr,
    n_indices,
    b_shape,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0)
    if index >= n_indices:
        return

    idx = tl.load(indices_ptr + index, tl.int32)
    if idx >= b_shape:
        return

    start_b = idx * k
    start_out = index * k

    for i in range(k):
        tl.store(output_ptr + start_out + i, tl.load(b_ptr + start_b + i))


@torch.fx.wrap
def kernel_wrapper(a, b):
    n_indices = a[0].numel()
    b_shape = b.shape[0]
    k = b.shape[1]
    output = torch.empty((n_indices, k), dtype=b.dtype, device=b.device)

    index_select_kernel[(1,)](
        indices_ptr=a[0],
        b_ptr=b,
        output_ptr=output,
        n_indices=n_indices,
        b_shape=b_shape,
        k=k,
    )

    return (a[1], output)

def replacement_func():
    return kernel_wrapper