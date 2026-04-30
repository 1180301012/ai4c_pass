import torch
import triton
import triton.language as tl

@triton.jit
def avgpool_flatten_kernel(
    x_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused adaptive_avg_pool2d + flatten kernel.
    
    Performs 2D adaptive average pooling with output size 1x1,
    then flattens the result from [B, C, 1, 1] to [B, C].
    
    Input:  x, shape [B, C, H, W]
    Output: [B, C]
    """
    pid = tl.program_id(0)
    
    # Each program handles one channel across all batches
    # Total programs = B * C
    batch = pid // C
    ch = pid % C
    
    # Strides for input [B, C, H, W]
    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    
    # Accumulator for average
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute average for this batch and channel
    # Sum over all H*W positions
    for h in range(H):
        for w in range(W):
            offset = batch * stride_b + ch * stride_c + h * stride_h + w
            offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (batch + 1) * stride_b
            
            val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            acc = acc + val
    
    # Average: divide by H*W
    avg_val = acc / (H * W)
    
    # Output is [B, C], so output offset is batch * C + ch
    out_offset = batch * C + ch
    offsets_out = out_offset + tl.arange(0, BLOCK_SIZE)
    mask_out = offsets_out < B * C
    
    # Store result
    tl.store(out_ptr + offsets_out, avg_val, mask=mask_out)


@torch.fx.wrap
def avgpool_flatten_wrapper(x):
    """
    Fused adaptive_avg_pool2d(output_size=1) + flatten(1, -1)
    
    Args:
        x: Input tensor, shape [B, C, H, W]
    
    Returns:
        Pooled and flattened tensor, shape [B, C]
    """
    B, C, H, W = x.shape
    
    # Output shape [B, C]
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    
    # Grid: one program per (batch, channel) pair
    grid = (B * C,)
    BLOCK_SIZE = 1024
    
    avgpool_flatten_kernel[grid](
        x, out,
        B, C, H, W,
        BLOCK_SIZE
    )
    
    return out


def pattern(x):
    """
    Pattern: adaptive_avg_pool2d(x, 1) followed by flatten(1, -1)
    
    This is a common pattern in SE blocks where global average pooling
    is followed by a fully connected layer.
    The flatten converts [B, C, 1, 1] to [B, C].
    """
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    out = pooled.flatten(1, -1)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return avgpool_flatten_wrapper