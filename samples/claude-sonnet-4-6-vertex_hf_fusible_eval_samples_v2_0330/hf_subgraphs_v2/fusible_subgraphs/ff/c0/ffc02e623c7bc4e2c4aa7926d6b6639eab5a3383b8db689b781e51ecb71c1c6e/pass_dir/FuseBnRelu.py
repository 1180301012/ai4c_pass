import torch
import triton
import triton.language as tl


@triton.jit
def triton_relu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


@torch.fx.wrap
def triton_relu_inplace(x):
    n = x.numel()
    BLOCK = 1024
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    triton_relu_kernel[((n + BLOCK - 1) // BLOCK,)](x, out, n, BLOCK_SIZE=BLOCK)
    return out.view(x.shape)


# ── Pattern: torch.relu_(x) — Dynamo captures inplace relu as torch.relu_ ────
def pattern(x):
    return torch.relu_(x)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_relu_inplace