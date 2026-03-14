import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    # Match the exact computation from the model
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel with better memory access patterns
@triton.jit
def optimized_fused_kernel_better(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel with better memory coalescing"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with better memory access pattern
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Parallel computation (simple but efficient)
    out = x * 1.0  # Identity operation here - actual optimization is in the wrapper
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

# Alternative fused computation with different reduction order
@torch.fx.wrap
def optimized_fusion_different_order(in_0):
    """Alternative fusion strategy with different reduction ordering
    
    Try different reduction order for potentially better GPU performance:
    1. First mean over spatial dimensions (width1, width2)
    2. Then mean over channels and height
    """
    batch_size, channels, height, width1, width2 = in_0.shape
    
    # Strategy: Mean over spatial dimensions first, then expand and reduce
    mean_spatial = in_0.mean(dim=[3, 4], keepdim=True)  # [batch, channels, height, 1, 1]
    
    # Now need to reduce channels dimension
    result = mean_spatial.mean(dim=[1], keepdim=True)   # [batch, 1, height, 1, 1] -> need to reshape
    
    # Reshape to match expected [batch, height, 1, 1]
    return result.transpose(1, 2).squeeze(1).unsqueeze(-1)

# Replacement function
def replacement_func():
    return optimized_fusion_different_order