import torch
import triton
import triton.language as tl

# Pattern matching function - try simplest possible pattern
def pattern(x):
    # Just match mean operation
    y = x.mean((2, 3), keepdim=True)
    return y

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def mean_kernel(
    x_ptr, out_ptr,
    N, C, H, W, HW,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    total_acc = 0.0
    base_offset = n * stride_n + c * stride_c
    
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h = offs // W
        w = offs % W
        idx = base_offset + h * stride_h + w * stride_w
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        total_acc += tl.sum(tl.where(mask, x, 0.0))
    
    mean_val = total_acc / HW
    tl.store(out_ptr + n * C + c, mean_val)


@torch.fx.wrap
def fused_mean(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    mean_kernel[grid](
        x, out,
        N, C, H, W, HW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
    )
    return out


def replacement_func():
    return fused_mean