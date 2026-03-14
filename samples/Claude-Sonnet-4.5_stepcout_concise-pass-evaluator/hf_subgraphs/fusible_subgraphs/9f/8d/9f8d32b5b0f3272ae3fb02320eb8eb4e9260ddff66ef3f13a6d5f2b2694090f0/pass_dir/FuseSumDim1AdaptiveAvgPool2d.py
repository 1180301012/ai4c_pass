import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_sum_adaptive_pool_kernel(
    input_ptr,
    output_ptr,
    B, C, H, W,
    stride_b, stride_d1, stride_c, stride_h, stride_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (B, C)
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Each program computes output[b_idx, c_idx, 0, 0]
    # We need to sum over dim=1 (size 2) and average over H*W
    
    accumulator = 0.0
    
    # Sum over dim=1 (size 2) - unrolled for better performance
    # Process spatial dimensions in 2D blocks for memory coalescing
    for d1 in range(2):
        base_offset = b_idx * stride_b + d1 * stride_d1 + c_idx * stride_c
        
        for h_base in range(0, H, BLOCK_H):
            for w_base in range(0, W, BLOCK_W):
                h_offs = h_base + tl.arange(0, BLOCK_H)
                w_offs = w_base + tl.arange(0, BLOCK_W)
                
                h_mask = h_offs < H
                w_mask = w_offs < W
                
                # 2D addressing
                h_idx = h_offs[:, None]
                w_idx = w_offs[None, :]
                
                indices = base_offset + h_idx * stride_h + w_idx * stride_w
                mask = h_mask[:, None] & w_mask[None, :]
                
                vals = tl.load(input_ptr + indices, mask=mask, other=0.0)
                accumulator += tl.sum(vals)
    
    # Average over H*W
    result = accumulator / (H * W)
    
    # output shape: [B, C, 1, 1]
    output_idx = b_idx * C + c_idx
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_sum_adaptive_pool(in_0):
    B, D1, C, H, W = in_0.shape
    assert D1 == 2, "This kernel expects dimension 1 to have size 2"
    
    output = torch.empty((B, C, 1, 1), device=in_0.device, dtype=in_0.dtype)
    
    grid = (B, C)
    
    # Choose block size based on spatial dimensions
    if H <= 16 and W <= 16:
        BLOCK_H, BLOCK_W = 16, 16
    elif H <= 32 and W <= 32:
        BLOCK_H, BLOCK_W = 32, 32
    else:
        BLOCK_H, BLOCK_W = 16, 16
    
    fused_sum_adaptive_pool_kernel[grid](
        in_0,
        output,
        B, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output

def replacement_func():
    return fused_sum_adaptive_pool