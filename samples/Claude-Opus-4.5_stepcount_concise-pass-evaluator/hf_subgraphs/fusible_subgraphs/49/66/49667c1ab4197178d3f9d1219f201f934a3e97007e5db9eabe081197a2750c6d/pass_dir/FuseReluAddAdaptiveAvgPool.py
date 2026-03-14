import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Pattern to match: AdaptiveAvgPool2d(1)
    """
    out = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    return out


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['spatial_size'],
)
@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    NC,  # N * C
    spatial_size,  # H * W
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for Global Average Pooling
    Each program handles one (batch, channel) pair
    """
    pid = tl.program_id(0)
    
    base_offset = pid * spatial_size
    
    # Accumulator for sum
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for start in range(0, spatial_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        acc += tl.where(mask, x_vals, 0.0)
    
    total_sum = tl.sum(acc, axis=0)
    avg = total_sum / spatial_size
    
    tl.store(out_ptr + pid, avg)


@torch.fx.wrap
def global_avg_pool(x):
    """
    Wrapper function for the global average pooling kernel
    """
    N, C, H, W = x.shape
    spatial_size = H * W
    NC = N * C
    
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    x = x.contiguous()
    
    global_avg_pool_kernel[(NC,)](
        x,
        out,
        NC,
        spatial_size,
    )
    
    return out


def replacement_func():
    return global_avg_pool