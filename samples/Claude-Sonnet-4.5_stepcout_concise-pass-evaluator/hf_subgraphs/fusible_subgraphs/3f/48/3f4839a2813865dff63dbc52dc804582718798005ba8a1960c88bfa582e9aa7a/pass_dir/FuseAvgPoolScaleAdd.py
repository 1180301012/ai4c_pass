import torch
import triton
import triton.language as tl

def pattern(in_0, in_2):
    """
    Pattern matching the computation:
    - avg_pool2d on in_2
    - subtract original
    - scale with in_0 (unsqueezed)
    - add back to original
    """
    tmp_0 = in_0
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    tmp_3 = tmp_2 - in_2
    tmp_4 = tmp_0.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 * tmp_3
    tmp_7 = in_2 + tmp_6
    return tmp_7

def replacement_args(in_0, in_2):
    return (in_0, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def fused_avgpool_scale_add_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. avg_pool2d with kernel=3, stride=1, padding=1
    2. subtract original
    3. multiply by scale (broadcasted from [C] to [B, C, H, W])
    4. add back to original
    """
    # Each program handles one position in the output
    pid = tl.program_id(0)
    
    # Total elements
    total_elements = B * C * H * W
    
    # Calculate batch, channel, height, width indices
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose linear index
    w = idx % W
    h = (idx // W) % H
    c = (idx // (W * H)) % C
    b = idx // (C * W * H)
    
    # Load original value
    input_idx = b * (C * H * W) + c * (H * W) + h * W + w
    original = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Compute average pooling (3x3 kernel with padding=1, stride=1)
    # Initialize with proper types (vector)
    pool_sum = tl.zeros_like(original)
    count = tl.zeros_like(original)
    
    # Unroll the pooling loop for better performance
    for dh in range(-1, 2):
        for dw in range(-1, 2):
            nh = h + dh
            nw = w + dw
            
            # Check bounds
            valid = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W) & mask
            
            pool_idx = b * (C * H * W) + c * (H * W) + nh * W + nw
            val = tl.load(input_ptr + pool_idx, mask=valid, other=0.0)
            
            pool_sum += tl.where(valid, val, 0.0)
            count += tl.where(valid, 1.0, 0.0)
    
    pooled = pool_sum / count
    
    # Subtract original
    diff = pooled - original
    
    # Load scale factor (broadcasted from [C])
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    
    # Multiply and add back
    result = original + scale * diff
    
    # Store result
    tl.store(output_ptr + input_idx, result, mask=mask)

@torch.fx.wrap
def fused_avgpool_scale_add(in_0, in_2):
    """
    Wrapper function that launches the fused kernel with autotuning
    """
    B, C, H, W = in_2.shape
    
    # Allocate output
    out = torch.empty_like(in_2)
    
    # Launch kernel with autotuning - grid will be dynamically computed
    total_elements = B * C * H * W
    
    # The autotuner will select the best BLOCK_SIZE
    def grid(meta):
        return ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_avgpool_scale_add_kernel[grid](
        in_2,
        in_0,
        out,
        B, C, H, W,
    )
    
    return out

def replacement_func():
    return fused_avgpool_scale_add