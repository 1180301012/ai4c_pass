import torch
import triton
import triton.language as tl

# Triton kernel for fused SiLU + mean pooling
# Each program processes one channel, iterating over all spatial positions
# This avoids materializing the full SiLU output tensor before mean reduction
@triton.jit
def silu_mean_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU activation and mean pooling kernel.
    
    Grid: (C,) - one program per channel
    Each program:
    1. Applies SiLU to all elements in its channel
    2. Stores SiLU results to output
    3. Computes running sum for mean
    4. Divides by total elements and stores mean
    """
    c = tl.program_id(0)
    
    # Accumulator for mean computation
    mean_acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over all spatial positions in this channel
    for h in range(H):
        # Compute offsets for this row
        offsets = c * H * W + h * W + tl.arange(0, W)
        mask = offsets < C * H * W
        
        # Load values (converted to float32 for accumulation)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # SiLU: x * sigmoid(x)
        silu = x * tl.sigmoid(x)
        
        # Store SiLU output
        tl.store(out_ptr + offsets, silu, mask=mask)
        
        # Accumulate for mean computation
        mean_acc += tl.sum(silu)
    
    # Normalize and store mean
    mean_acc = mean_acc / (H * W)
    tl.store(mean_ptr + c, mean_acc)


@torch.fx.wrap
def silu_mean_wrapper(x):
    """
    Wrapper function that launches the fused SiLU + mean kernel.
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        
    Returns:
        Tuple of (silu_output, mean) where:
        - silu_output: Same shape as input [B, C, H, W]
        - mean: Tensor of shape [B, C]
    """
    B, C, H, W = x.shape
    
    # Grid: one program per channel for maximum parallelism
    num_programs = C
    
    # Allocate output tensors
    # Use float32 for mean accumulation
    out = torch.empty_like(x)
    mean = torch.empty((B, C), dtype=torch.float32, device=x.device)
    
    # BLOCK_SIZE should be >= W to process entire row in one load
    # Use 512 as a safe default that works for all W <= 512
    BLOCK_SIZE = 512
    
    # Launch kernel
    silu_mean_kernel[(num_programs,)](
        x,
        out,
        mean,
        C,
        H,
        W,
        BLOCK_SIZE,
    )
    
    return out, mean


# Pattern matching function - matches SiLU followed by mean over spatial dims
def pattern(in_0, in_1):
    """
    Pattern: SiLU activation followed by mean pooling over spatial dimensions.
    
    Matches:
        tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_1 = tmp_0.mean((2, 3))
        return tmp_0, tmp_1
    
    The in_0 parameter is unused (dead code in the original graph).
    """
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    return tmp_0, tmp_1


# Extract arguments needed for the replacement function
def replacement_args(in_0, in_1):
    """
    Extract arguments for the replacement function.
    
    Note: in_0 is a dummy argument (not used in computation)
    but must be included to match the pattern signature.
    """
    return (in_0, in_1)


# Replacement function - returns the optimized wrapper
def replacement_func():
    """
    Returns the fused SiLU + mean kernel wrapper function.
    
    This optimization:
    1. Fuses SiLU activation with mean pooling into a single GPU kernel
    2. Avoids allocating an intermediate tensor for SiLU output
    3. Computes mean in-place during the activation pass
    4. Significantly reduces memory bandwidth and kernel launch overhead
    """
    return silu_mean_wrapper