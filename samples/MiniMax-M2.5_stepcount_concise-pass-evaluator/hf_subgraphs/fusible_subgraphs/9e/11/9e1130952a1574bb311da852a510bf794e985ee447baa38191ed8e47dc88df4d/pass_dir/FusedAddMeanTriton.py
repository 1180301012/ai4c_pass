import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_K': 512}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_K': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_K': 2048}, num_stages=3, num_warps=4),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_add_mean_triton_kernel(
    in4_ptr,
    in5_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for fused add + mean over spatial dimensions.
    
    Each program processes one (batch, channel) position and reduces
    over all spatial positions (H*W).
    
    Uses BLOCK_SIZE_K to control the reduction block size.
    """
    # Program ID for batch and channel
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Calculate starting offset in the flattened tensor
    # Layout: [B, C, H, W] flattened
    base_offset = batch_idx * C * spatial_size + channel_idx * spatial_size
    
    # Initialize accumulator
    sum_val = 0.0
    
    # Process in blocks
    for k in range(0, spatial_size, BLOCK_SIZE_K):
        offsets = base_offset + k + tl.arange(0, BLOCK_SIZE_K)
        mask = k + tl.arange(0, BLOCK_SIZE_K) < spatial_size
        
        # Load from both tensors
        v4 = tl.load(in4_ptr + offsets, mask=mask, other=0.0)
        v5 = tl.load(in5_ptr + offsets, mask=mask, other=0.0)
        
        # Add and accumulate
        sum_val += tl.sum(v4 + v5, axis=0)
    
    # Compute mean
    mean_val = sum_val / spatial_size
    
    # Store result
    out_idx = batch_idx * C + channel_idx
    tl.store(output_ptr + out_idx, mean_val)


@torch.fx.wrap
def fused_add_mean_triton(in4: torch.Tensor, in5: torch.Tensor) -> torch.Tensor:
    """
    Fused add + mean using Triton kernel.
    Computes: (in4 + in5).mean(dim=(2,3))
    """
    B, C, H, W = in4.shape
    spatial_size = H * W
    
    # Create output tensor
    output = torch.empty((B, C), device=in4.device, dtype=in4.dtype)
    
    # Launch kernel - one program per (batch, channel) position
    grid = (B * C,)
    
    fused_add_mean_triton_kernel[grid](
        in4,
        in5,
        output,
        B,
        C,
        spatial_size,
    )
    
    return output


def pattern(in_4, in_5):
    """
    Pattern: tmp_4 = in_5 + in_4; tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


def replacement_func():
    return fused_add_mean_triton