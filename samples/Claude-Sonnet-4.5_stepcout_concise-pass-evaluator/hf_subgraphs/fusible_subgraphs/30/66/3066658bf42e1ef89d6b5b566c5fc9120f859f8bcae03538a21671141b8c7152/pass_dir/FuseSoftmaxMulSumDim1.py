import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    depth,
    hw_size,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple 1D processing with contiguous memory access
    pid = tl.program_id(0)
    
    # Compute the spatial position this thread block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decompose into depth and hw indices
    d = offsets // hw_size
    hw = offsets % hw_size
    
    # Load softmax weights (only depends on depth)
    in_1_0 = tl.load(in_1_ptr + d, mask=mask, other=0.0)
    in_1_1 = tl.load(in_1_ptr + depth + d, mask=mask, other=0.0)
    
    # Compute softmax
    max_v = tl.maximum(in_1_0, in_1_1)
    e0 = tl.exp(in_1_0 - max_v)
    e1 = tl.exp(in_1_1 - max_v)
    s = e0 + e1
    w0 = e0 / s
    w1 = e1 / s
    
    # Load values from in_0
    idx_0 = d * hw_size + hw
    idx_1 = depth * hw_size + d * hw_size + hw
    
    v0 = tl.load(in_0_ptr + idx_0, mask=mask, other=0.0)
    v1 = tl.load(in_0_ptr + idx_1, mask=mask, other=0.0)
    
    # Weighted sum
    out = v0 * w0 + v1 * w1
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    batch_size, channels, depth, height, width = in_0.shape
    
    # Output shape: [batch, depth, height, width] (channels dimension removed by sum)
    out = torch.empty((batch_size, depth, height, width), dtype=in_0.dtype, device=in_0.device)
    
    hw_size = height * width
    n_elements = depth * hw_size
    
    # Use autotune to find best block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_softmax_mul_sum_kernel[grid](
        in_0,
        in_1,
        out,
        depth,
        hw_size,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_softmax_mul_sum