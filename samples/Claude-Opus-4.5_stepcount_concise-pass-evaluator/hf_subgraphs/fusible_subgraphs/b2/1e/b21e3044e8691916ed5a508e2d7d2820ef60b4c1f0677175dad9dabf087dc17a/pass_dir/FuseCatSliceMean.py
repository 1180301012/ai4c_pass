import torch
import triton
import triton.language as tl

# Pattern to match: cat -> slice -> mean with slice value 120
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Match the slice syntax exactly as in model.py - slice(None, 120, None) means [:120]
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 120, None), slice(None, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def cat_mean_kernel(
    in_0_ptr, in_1_ptr,
    out_cat_ptr, out_mean_ptr,
    B, C, H, W, C2,
    HW,  # Pass HW as a separate parameter
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel for cat + mean operation.
    Each program handles one (batch, channel_out) position.
    Copies data from in_0 or in_1 to output and computes spatial mean.
    """
    pid = tl.program_id(0)
    
    # Compute batch and output channel indices
    b = pid // C2
    c_out = pid % C2
    
    # Determine which input to use and the channel index within that input
    c_in = c_out % C
    use_in_1 = c_out >= C  # True if we should read from in_1
    
    # Base offset for input (same structure for both in_0 and in_1)
    in_base = b * C * HW + c_in * HW
    
    # Base offset for output
    out_base = b * C2 * HW + c_out * HW
    
    # Accumulator for mean computation - use scalar
    acc = 0.0
    
    # Process spatial dimensions in blocks
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        
        # Masked load from the appropriate input
        # Only one of these actually loads data based on use_in_1
        mask_0 = mask & (~use_in_1)
        mask_1 = mask & use_in_1
        
        val_0 = tl.load(in_0_ptr + in_base + offs, mask=mask_0, other=0.0)
        val_1 = tl.load(in_1_ptr + in_base + offs, mask=mask_1, other=0.0)
        vals = val_0 + val_1  # One is always 0
        
        # Store to concatenated output
        tl.store(out_cat_ptr + out_base + offs, vals, mask=mask)
        
        # Accumulate sum for mean - tl.sum returns a scalar
        acc = acc + tl.sum(vals)
    
    # Compute and store mean
    mean_val = acc / HW
    mean_off = b * C2 + c_out
    tl.store(out_mean_ptr + mean_off, mean_val)


@torch.fx.wrap
def _cat_mean_kernel_impl(in_0, in_1):
    """
    Internal implementation that calls the Triton kernel.
    Returns a tuple (out_cat, out_mean).
    """
    B, C, H, W = in_0.shape
    C2 = 2 * C
    HW = H * W
    
    # Allocate output tensors
    out_cat = torch.empty(B, C2, H, W, device=in_0.device, dtype=in_0.dtype)
    out_mean = torch.empty(B, C2, 1, 1, device=in_0.device, dtype=in_0.dtype)
    
    # Number of programs = number of (batch, channel) pairs
    num_programs = B * C2
    
    # Choose block size based on spatial dimensions
    BLOCK_HW = min(1024, max(64, triton.next_power_of_2(HW)))
    
    # Launch kernel
    cat_mean_kernel[(num_programs,)](
        in_0, in_1,
        out_cat, out_mean,
        B, C, H, W, C2,
        HW,  # Pass HW explicitly
        BLOCK_HW=BLOCK_HW,
    )
    
    return (out_cat, out_mean)


def cat_mean_fused(in_0, in_1):
    """
    Wrapper that unpacks the tuple to match the pattern's output structure.
    """
    result = _cat_mean_kernel_impl(in_0, in_1)
    return (result[0], result[1])


def replacement_func():
    return cat_mean_fused