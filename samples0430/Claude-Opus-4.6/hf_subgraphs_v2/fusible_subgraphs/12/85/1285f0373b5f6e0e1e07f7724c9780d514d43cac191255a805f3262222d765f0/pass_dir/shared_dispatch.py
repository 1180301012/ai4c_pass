import torch
import triton
import triton.language as tl


@triton.jit
def cat_dim1_kernel(
    a_ptr, b_ptr, out_ptr,
    a_cols, b_cols, out_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    # Copy from a
    offsets = tl.arange(0, BLOCK)
    mask_a = offsets < a_cols
    a_vals = tl.load(a_ptr + row * a_cols + offsets, mask=mask_a)
    tl.store(out_ptr + row * out_cols + offsets, a_vals, mask=mask_a)

    # Copy from b
    mask_b = offsets < b_cols
    b_vals = tl.load(b_ptr + row * b_cols + offsets, mask=mask_b)
    tl.store(out_ptr + row * out_cols + a_cols + offsets, b_vals, mask=mask_b)


@triton.jit
def fill_ones_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, tl.full([BLOCK], 1.0, dtype=tl.float32), mask=mask)


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "cat_dim1":
        a = args[0]
        b = args[1]
        rows = a.shape[0]
        a_cols = a.shape[1]
        b_cols = b.shape[1]
        out_cols = a_cols + b_cols
        out = torch.empty((rows, out_cols), dtype=a.dtype, device=a.device)
        cat_dim1_kernel[(rows,)](a, b, out, a_cols, b_cols, out_cols, BLOCK=1024)
        return out
    elif route == "ones":
        size = args[0]
        out = torch.ones((size,), dtype=torch.float32, device='cuda')
        return out
    else:
        size = args[0]
        return torch.ones((size,), dtype=torch.float32, device='cuda')