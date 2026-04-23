import torch
import triton
import triton.language as tl


@triton.jit
def silu_mean_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    n_channels,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one channel
    ch_idx = tl.program_id(0)
    
    # Check if this channel is valid
    if ch_idx >= n_channels:
        return
    
    # Initialize accumulator for mean
    accum = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over spatial dimensions
    for h_idx in range(H):
        for w_idx in range(W):
            # Calculate offset for this element
            # Tensor shape: [1, n_channels, H, W]
            # Stride: [n_channels*H*W, H*W, W, 1]
            offset = ch_idx * H * W + h_idx * W + w_idx
            
            # Load value
            x = tl.load(input_ptr + offset).to(tl.float32)
            
            # Apply SiLU: x * sigmoid(x)
            sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
            silu_x = x * sigmoid_x
            
            # Store silu output
            tl.store(output_ptr + offset, silu_x.to(input_ptr.dtype))
            
            # Accumulate for mean
            accum = accum + silu_x
    
    # Compute mean
    mean_val = accum / (H * W)
    
    # Store mean result
    tl.store(mean_ptr + ch_idx, mean_val)


@torch.fx.wrap
def silu_mean_fused(in_1):
    """
    Fused SiLU + mean over spatial dimensions (2, 3).
    Returns (silu_output, mean_result) to match pattern return structure.
    """
    # Main optimization: fused silu + mean
    batch_size, n_channels, H, W = in_1.shape
    assert batch_size == 1, "Batch size must be 1 for this optimization"
    
    # Output tensor for silu result
    silu_out = torch.empty_like(in_1)
    
    # Output tensor for mean result
    mean_out = torch.empty((n_channels,), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel - one program per channel
    BLOCK_SIZE = 1
    
    silu_mean_kernel[(n_channels,)](
        in_1,
        silu_out,
        mean_out,
        n_channels,
        H,
        W,
        BLOCK_SIZE,
    )
    
    # Apply view to reshape mean output to [1, 1, n_channels]
    mean_view = mean_out.view(1, 1, -1)
    
    return silu_out, mean_view


def pattern(x, y):
    """
    Match the silu + mean + view pattern.
    Uses generic names to match any argument names.
    """
    tmp_0 = torch.nn.functional.silu(y, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_0, tmp_4


def replacement_args(x, y):
    # Extract arguments needed for replacement
    return (y,)


def replacement_func():
    return silu_mean_fused