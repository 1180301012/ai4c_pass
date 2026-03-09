import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=16),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_add_mean_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_0_ptr, out_1_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    - out_0 = in_0 + in_1 + in_2 (element-wise addition)
    - out_1 = mean(out_0, dim=(2, 3), keepdim=True)
    """
    # Get position in the batch/channel dimensions
    pid = tl.program_id(0)
    
    # Calculate total elements per batch and channel
    hw = H * W
    
    # Calculate which (batch, channel) this program computes
    batch_stride = C * hw
    
    # Each program handles one (batch, channel) position
    base_idx = pid * batch_stride
    
    # Initialize accumulators for mean computation
    sum_vals = tl.zeros((1,), tl.float32)
    
    # Process all spatial positions for this (batch, channel)
    for offset in range(0, hw, BLOCK_SIZE):
        # Create offsets for this block
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < hw
        
        # Calculate actual memory offsets
        base_offs = base_idx + offs
        
        # Load inputs
        x0 = tl.load(in_0_ptr + base_offs, mask=mask, other=0.0)
        x1 = tl.load(in_1_ptr + base_offs, mask=mask, other=0.0)
        x2 = tl.load(in_2_ptr + base_offs, mask=mask, other=0.0)
        
        # Compute sum
        s = x0 + x1 + x2
        
        # Store the sum result
        tl.store(out_0_ptr + base_offs, s, mask=mask)
        
        # Accumulate for mean
        sum_vals += tl.sum(s, axis=0)
    
    # Compute mean: divide by total spatial elements
    mean_val = sum_vals / tl.cast(hw, tl.float32)
    
    # Store mean
    tl.store(out_1_ptr + pid, mean_val)


def pattern(in_0, in_1):
    """
    Pattern: Element-wise addition of 2 tensors followed by mean reduction.
    This matches patterns like: tmp_0 = 0 + in_1; tmp_0 += in_0; tmp_1 = tmp_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
    """
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1, None)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2=None):
    """
    Wrapper function that launches the fused add + mean kernel.
    """
    # Get dimensions
    N, C, H, W = in_0.shape
    
    # Determine number of programs (one per N*C)
    num_programs = N * C
    
    # Allocate output tensors
    out_0 = torch.empty_like(in_0)
    out_1 = torch.empty((N, C, 1, 1), dtype=torch.float32, device=in_0.device)
    
    # Handle different input configurations
    if in_2 is None:
        in_2 = torch.zeros_like(in_0)
    
    # Launch kernel
    grid = (num_programs,)
    
    fused_add_mean_kernel[grid](
        in_0, in_1, in_2,
        out_0, out_1,
        N, C, H, W,
        BLOCK_SIZE=1024,
    )
    
    return (out_0, out_1)


def replacement_func():
    return kernel_wrapper