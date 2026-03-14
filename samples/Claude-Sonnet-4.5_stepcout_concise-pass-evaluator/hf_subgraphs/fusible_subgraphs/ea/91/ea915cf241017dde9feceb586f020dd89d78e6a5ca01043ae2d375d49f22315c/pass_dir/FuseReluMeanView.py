import torch
import triton
import triton.language as tl

def pattern(relu_in):
    """Pattern to match: mean over spatial dims -> view
    Note: relu_in is the output from relu, we match from there"""
    mean_out = relu_in.mean((2, 3))
    view_out = mean_out.view(1, 1, -1)
    return view_out

def replacement_args(relu_in):
    """Extract arguments for replacement"""
    return (relu_in,)

@triton.jit
def fused_mean_kernel(
    input_ptr,
    output_ptr,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Minimal overhead kernel for small reductions.
    """
    pid = tl.program_id(0)
    b_idx = pid // C
    c_idx = pid % C
    base_offset = (b_idx * C + c_idx) * H * W
    spatial_size = H * W
    
    acc = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, spatial_size, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < spatial_size
        vals = tl.load(input_ptr + base_offset + idx, mask=mask, other=0.0)
        acc += tl.sum(vals)
    
    mean_val = acc / spatial_size
    tl.store(output_ptr + b_idx * C + c_idx, mean_val)

@torch.fx.wrap
def fused_mean_view(x):
    """
    Minimalist implementation with single warp.
    """
    B, C, H, W = x.shape
    spatial_size = H * W
    
    output = torch.empty((B, C), dtype=x.dtype, device=x.device)
    
    # Use minimal resources - single warp for small reductions
    grid = (B * C,)
    if spatial_size <= 64:
        BLOCK_SIZE = 64
        num_warps = 1
    elif spatial_size <= 128:
        BLOCK_SIZE = 128
        num_warps = 1
    elif spatial_size <= 512:
        BLOCK_SIZE = 512
        num_warps = 2
    else:
        BLOCK_SIZE = 1024
        num_warps = 2
    
    fused_mean_kernel[grid](
        x,
        output,
        B, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output.view(1, 1, -1)

def replacement_func():
    """Return the optimized implementation"""
    return fused_mean_view