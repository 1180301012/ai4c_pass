import torch
import triton
import triton.language as tl


def pattern(x):
    # Match mean(dim=-2, keepdim=True)
    # Input shape: [B, 4096, 256], Output shape: [B, 1, 256]
    result = x.mean(dim=-2, keepdim=True)
    return result


def replacement_args(x):
    return (x,)


# Autotuned Triton kernel for mean computation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['H'],
)
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (B, W)
    pid_b = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    output_offset = pid_b * W + pid_w
    base_offset = pid_b * (H * W) + pid_w
    
    acc = 0.0
    for i in range(0, H, BLOCK_SIZE):
        offs_h = i + tl.arange(0, BLOCK_SIZE)
        mask = offs_h < H
        ptrs = base_offset + offs_h * W
        vals = tl.load(input_ptr + ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    mean_val = acc / tl.cast(H, tl.float32)
    tl.store(output_ptr + output_offset, mean_val)


@torch.fx.wrap
def mean_kernel_wrapper(x):
    B, H, W = x.shape
    output = torch.empty((B, 1, W), dtype=x.dtype, device=x.device)
    
    mean_kernel[(B, W)](
        x,
        output,
        B,
        H,
        W,
    )
    
    return output


def replacement_func():
    return mean_kernel_wrapper