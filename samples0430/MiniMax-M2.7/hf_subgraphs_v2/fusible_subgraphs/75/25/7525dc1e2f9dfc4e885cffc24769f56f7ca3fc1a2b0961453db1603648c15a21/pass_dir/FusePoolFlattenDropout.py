import torch
import triton
import triton.language as tl

# Fixed block size using constexpr
BLOCK_SIZE = tl.constexpr(64)

@triton.jit
def fused_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    B, C, H, W,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    N_elements: tl.constexpr,
):
    """
    Fused adaptive_avg_pool2d(output_size=1) + flatten kernel.
    Input: [B, C, H, W]
    Output: [B * C] (flattened to 1D)
    
    Adaptive avg pool with output_size=1 computes the mean over all spatial dims.
    This can be expressed as: out[b,c] = sum over h,w of input[b,c,h,w] / (H*W)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements
    
    # Calculate b, c from flat offset
    # output[b,c] = input[b, c, :, :].mean() -> sum / (H*W)
    b = offsets // C
    c = offsets % C
    
    # Compute base offset for this batch and channel
    base_offset = b * input_stride_b + c * input_stride_c
    
    # Sum all H*W elements for this channel using a loop
    total = 0.0
    for h in range(H):
        for w in range(W):
            offset = base_offset + h * input_stride_h + w * input_stride_w
            total = total + tl.load(input_ptr + offset)
    
    # Compute mean
    out_val = total / (H * W)
    
    # Store output (flattened to 1D)
    tl.store(output_ptr + offsets, out_val, mask=mask)


def pattern(tmp_4):
    """
    Match the pattern: adaptive_avg_pool2d + flatten + dropout(p=0.0)
    This pattern is common in models where global pooling is followed by classification.
    """
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(tmp_4):
    """Extract arguments needed for the replacement kernel."""
    return (tmp_4,)


@torch.fx.wrap
def fused_pool_flatten_wrapper(tmp_4):
    """
    Fused implementation that combines:
    1. Adaptive average pooling (output_size=1)
    2. Flatten to 1D
    3. Dropout with p=0.0 (no-op, eliminated)
    """
    B, C, H, W = tmp_4.shape
    N_elements = B * C
    
    # Allocate output tensor: [B*C]
    out = torch.empty((B * C,), dtype=tmp_4.dtype, device=tmp_4.device)
    
    # Get input strides
    input_stride_b = C * H * W
    input_stride_c = H * W
    input_stride_h = W
    input_stride_w = 1
    
    # Calculate grid
    BLOCK = 1024
    num_programs = (N_elements + BLOCK - 1) // BLOCK
    
    # Launch kernel
    grid = (num_programs,)
    
    fused_pool_flatten_kernel[grid](
        tmp_4, out,
        B, C, H, W,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        N_elements
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_pool_flatten_wrapper