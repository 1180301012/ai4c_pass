import torch
import triton
import triton.language as tl


# Autotune configurations optimized for different tensor sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096, 'num_warps': 8}, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_permute_kernel(
    input_ptr, output_ptr,
    n_elements,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized permute: [B, C, H, W] -> [B, H, W, C]
    Uses contiguous memory access pattern for better coalescing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    HW = H * W
    b_idx = offsets // (HW * C)
    remaining = offsets % (HW * C)
    hw_idx = remaining // C
    c_idx = remaining % C
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # [B, C, H, W] layout: b*C*H*W + c*H*W + h*W + w
    input_offsets = b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx
    val = tl.load(input_ptr + input_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    
    # Output is [B, H, W, C] layout: b*HW*C + hw*C + c
    output_offsets = b_idx * HW * C + hw_idx * C + c_idx
    tl.store(output_ptr + output_offsets, val, mask=mask, eviction_policy="evict_last")


def pattern(x):
    """
    Match permute(0, 2, 3, 1)
    """
    return x.permute(0, 2, 3, 1)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def fused_permute_wrapper(x):
    B, C, H, W = x.shape
    n_elements = B * H * W * C
    output = torch.empty((B, H, W, C), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_permute_kernel[grid](x, output, n_elements, B, C, H, W)
    return output


def replacement_func():
    return fused_permute_wrapper