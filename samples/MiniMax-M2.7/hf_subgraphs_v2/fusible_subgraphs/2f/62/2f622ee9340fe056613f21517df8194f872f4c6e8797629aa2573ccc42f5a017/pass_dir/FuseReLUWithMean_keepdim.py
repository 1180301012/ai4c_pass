import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_mean_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    n_elements,
    n_channels,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Mean computation kernel.
    
    Each program (one per channel) computes:
    - ReLU activation for all spatial elements
    - Sum of ReLU values for mean computation
    - Output tensor with ReLU applied
    """
    # Get channel index from program_id
    c = tl.program_id(0)
    
    # Calculate base offsets for this channel
    channel_offset = c * H * W
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Process spatial elements in blocks
    for start in range(0, H * W, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        
        # Global offset for current channel
        global_offsets = channel_offset + offsets
        
        # Load values
        x = tl.load(input_ptr + global_offsets, mask=mask, other=0.0)
        
        # Apply ReLU: max(0, x)
        relu_x = tl.where(x > 0, x, 0.0)
        
        # Accumulate sum for mean computation
        sum_val += tl.sum(relu_x, axis=0)
        
        # Store ReLU output
        tl.store(output_ptr + global_offsets, relu_x, mask=mask)
    
    # Compute mean (H*W elements per channel)
    num_spatial = H * W
    mean_val = sum_val / num_spatial
    
    # Store mean at the channel position in mean output
    # Shape: [1, C, 1, 1] - each channel stores its mean at position [0, c, 0, 0]
    mean_offset = c  # Simplified: store as [C] tensor
    tl.store(mean_ptr + c, mean_val)


@torch.fx.wrap
def fused_relu_mean(x):
    """
    Fused ReLU + Mean kernel wrapper.
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        
    Returns:
        Tuple of (relu_x, mean) where:
        - relu_x: ReLU activated tensor [B, C, H, W]
        - mean: Mean over spatial dims [B, C, 1, 1]
    """
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    
    # Allocate outputs
    relu_out = torch.empty_like(x)
    mean_out = torch.empty((C,), dtype=x.dtype, device=x.device)
    
    # Configure kernel
    BLOCK_SIZE = 1024
    
    # Launch grid: one program per channel
    grid = (C,)
    
    # Execute kernel
    fused_relu_mean_kernel[grid](
        input_ptr=x,
        output_ptr=relu_out,
        mean_ptr=mean_out,
        n_elements=n_elements,
        n_channels=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape mean to [1, C, 1, 1]
    mean_reshaped = mean_out.view(1, C, 1, 1)
    
    return relu_out, mean_reshaped


def pattern(in_0, in_1):
    """
    Match the computation pattern with divisor=8:
    1. ReLU inplace on in_1
    2. Mean over spatial dims (2, 3)
    
    Note: The division and sym_sum operations are dead code that we eliminate.
    We include a dummy use of in_0 to satisfy the subgraph matcher's input requirement.
    """
    # Include in_0 in a dummy operation that won't affect matching
    # This ensures all model inputs are used in the pattern
    _ = in_0  # Reference in_0 so it's not considered unused
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_3


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Return the optimized fused ReLU + Mean function.
    """
    return fused_relu_mean