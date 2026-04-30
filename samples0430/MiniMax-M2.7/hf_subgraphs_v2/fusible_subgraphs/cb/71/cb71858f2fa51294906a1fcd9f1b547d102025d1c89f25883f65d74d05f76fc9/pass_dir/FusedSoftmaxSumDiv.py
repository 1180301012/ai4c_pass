import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_sum_div_kernel(
    in_ptr,
    out_ptr,
    n_slices: tl.constexpr,
    slice_size: tl.constexpr,
):
    """
    Fused kernel that computes: out = x / x.sum(dim=3, keepdim=True)
    
    For input shape [B, C, H, W], sum is along dim=3 (W dimension).
    Each program handles one "slice" of W elements.
    """
    pid = tl.program_id(0)
    
    # Compute starting offset for this slice
    base_offset = pid * slice_size
    
    # Load all elements in this slice
    offsets = base_offset + tl.arange(0, slice_size)
    mask = offsets < (n_slices * slice_size)
    
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum across the slice
    sum_val = tl.sum(x)
    
    # Normalize
    out = x / sum_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_softmax_sum_div(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of: out = x / x.sum(dim=3, keepdim=True)
    """
    # Input shape: [1, 2, 8, 8]
    # sum(dim=3) reduces along the last dimension (size 8)
    B, C, H, W = x.shape
    n_slices = B * C * H  # 1 * 2 * 8 = 16
    slice_size = W  # 8
    
    out = torch.empty_like(x)
    
    # One program per slice
    grid = (n_slices,)
    
    fused_softmax_sum_div_kernel[grid](
        in_ptr=x,
        out_ptr=out,
        n_slices=n_slices,
        slice_size=slice_size,
    )
    
    return out


def pattern(in_3):
    """
    Match the pattern: in_3.sum(dim=3, keepdim=True) followed by in_3 / sum_result
    """
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    # Extract arguments needed for the replacement
    return (in_3,)


def replacement_func():
    return fused_softmax_sum_div