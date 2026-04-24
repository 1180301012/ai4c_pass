import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def fused_relu_mean_kernel(
    x_ptr, out_relu_ptr, out_mean_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, c) pair; processes all HW elements at once
    pid = tl.program_id(0)
    offsets = pid * HW + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < HW

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    relu_val = tl.maximum(x, 0.0)

    # Store relu output
    tl.store(out_relu_ptr + offsets, relu_val.to(out_relu_ptr.dtype.element_ty), mask=mask)

    # Compute and store mean
    sum_val = tl.sum(relu_val, axis=0)
    mean_val = sum_val / HW
    tl.store(out_mean_ptr + pid, mean_val.to(out_mean_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_relu_mean(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out_relu = torch.empty_like(x)
    out_mean = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    fused_relu_mean_kernel[(NC,)](x, out_relu, out_mean, HW)
    return out_relu


def pattern(x):
    return torch.nn.functional.relu(x)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_relu_mean