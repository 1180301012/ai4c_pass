import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the LayerScale computation pattern:
    - avg_pool2d on in_2
    - Subtract in_2 from pooled result
    - Multiply by reshaped in_0 (layer scale 1)
    - Add to in_2
    - Reshape in_1 (layer scale 2) for output
    
    The pattern must return all values that appear in the model's return statement.
    
    The pattern must exactly mirror model.py operations including:
    - Using tmp_0, tmp_1 for copies of in_0, in_1
    - Using tmp_8 = tmp_1.unsqueeze(-1) (not in_1.unsqueeze)
    - Using tmp_9 = tmp_8.unsqueeze(-1) (not tmp_1)
    """
    # MUST match model.py exactly: tmp_0 = in_0, tmp_1 = in_1
    tmp_0 = in_0
    tmp_1 = in_1
    
    # avg_pool2d - must match exactly with all positional args
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    
    # Subtract
    tmp_3 = tmp_2 - in_2
    
    # First unsqueeze chain: tmp_4 = tmp_0.unsqueeze(-1), tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = tmp_0.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    
    # Multiply
    tmp_6 = tmp_5 * tmp_3
    
    # Add
    tmp_7 = in_2 + tmp_6
    
    # Second unsqueeze chain: tmp_8 = tmp_1.unsqueeze(-1), tmp_9 = tmp_8.unsqueeze(-1)
    # This is critical - model uses tmp_1, not in_1
    tmp_8 = tmp_1.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    
    return tmp_7, tmp_9


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2)


# Autotune configurations for different input sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def fused_layer_scale_kernel(
    in_ptr,          # Input tensor in_2
    scale1_ptr,      # Layer scale 1 (in_0)
    out_ptr,         # Output tensor tmp_7
    B: tl.constexpr,   # Batch size
    C: tl.constexpr,   # Channels
    H: tl.constexpr,   # Height
    W: tl.constexpr,   # Width
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for LayerScale operation:
    - Computes avg_pool2d via manual 3x3 convolution
    - Computes: in_2 + scale1 * (avg_pool(in_2) - in_2)
    
    This fuses 4 operations into 1 kernel:
    1. avg_pool2d (manual 3x3 sum with boundary handling)
    2. subtraction
    3. multiplication (broadcast)
    4. addition
    """
    # Get position for this thread block
    pid = tl.program_id(0)
    num_elements = B * C * H * W
    
    # Compute offsets
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    
    # Compute multi-dimensional indices
    w = offset % W
    h = (offset // W) % H
    c = (offset // (W * H)) % C
    b = offset // (W * H * C)
    
    # Load input value (in_2)
    in_val = tl.load(in_ptr + offset, mask=mask, other=0.0)
    
    # Load scale1 (broadcast by channel)
    scale1 = tl.load(scale1_ptr + c)
    
    # Compute average pooling manually - sum over 3x3 window
    # Using explicit boundary handling (same as avg_pool2d with padding=1)
    
    # Helper to compute clamped coordinate
    h_neighbor = h
    w_neighbor = w
    
    # Sum all 9 neighbors with boundary handling
    # Position (-1, -1) - upper left
    h1 = tl.where(h == 0, 0, h - 1)
    w1 = tl.where(w == 0, 0, w - 1)
    off1 = b * C * H * W + c * H * W + h1 * W + w1
    sum_val = tl.load(in_ptr + off1, mask=mask, other=0.0)
    
    # Position (0, -1) - left
    h2 = h
    w2 = tl.where(w == 0, 0, w - 1)
    off2 = b * C * H * W + c * H * W + h2 * W + w2
    sum_val = sum_val + tl.load(in_ptr + off2, mask=mask, other=0.0)
    
    # Position (1, -1) - lower left
    h3 = tl.where(h >= H - 1, H - 1, h + 1)
    w3 = tl.where(w == 0, 0, w - 1)
    off3 = b * C * H * W + c * H * W + h3 * W + w3
    sum_val = sum_val + tl.load(in_ptr + off3, mask=mask, other=0.0)
    
    # Position (-1, 0) - upper
    h4 = tl.where(h == 0, 0, h - 1)
    w4 = w
    off4 = b * C * H * W + c * H * W + h4 * W + w4
    sum_val = sum_val + tl.load(in_ptr + off4, mask=mask, other=0.0)
    
    # Position (0, 0) - center
    sum_val = sum_val + in_val
    
    # Position (1, 0) - lower
    h6 = tl.where(h >= H - 1, H - 1, h + 1)
    w6 = w
    off6 = b * C * H * W + c * H * W + h6 * W + w6
    sum_val = sum_val + tl.load(in_ptr + off6, mask=mask, other=0.0)
    
    # Position (-1, 1) - upper right
    h7 = tl.where(h == 0, 0, h - 1)
    w7 = tl.where(w >= W - 1, W - 1, w + 1)
    off7 = b * C * H * W + c * H * W + h7 * W + w7
    sum_val = sum_val + tl.load(in_ptr + off7, mask=mask, other=0.0)
    
    # Position (0, 1) - right
    h8 = h
    w8 = tl.where(w >= W - 1, W - 1, w + 1)
    off8 = b * C * H * W + c * H * W + h8 * W + w8
    sum_val = sum_val + tl.load(in_ptr + off8, mask=mask, other=0.0)
    
    # Position (1, 1) - lower right
    h9 = tl.where(h >= H - 1, H - 1, h + 1)
    w9 = tl.where(w >= W - 1, W - 1, w + 1)
    off9 = b * C * H * W + c * H * W + h9 * W + w9
    sum_val = sum_val + tl.load(in_ptr + off9, mask=mask, other=0.0)
    
    # Average (divide by 9)
    avg_val = sum_val / 9.0
    
    # Compute: in_val + scale1 * (avg_val - in_val)
    diff = avg_val - in_val
    scaled = scale1 * diff
    out_val = in_val + scaled
    
    # Store output
    tl.store(out_ptr + offset, out_val, mask=mask)


@torch.fx.wrap
def fused_layer_scale(scale1, scale2, in_2):
    """
    Fused layer scale operation.
    
    Args:
        scale1: Layer scale factor 1, shape [C]
        scale2: Layer scale factor 2, shape [C] 
        in_2: Input tensor, shape [B, C, H, W]
    
    Returns:
        out: in_2 + scale1 * (avg_pool2d(in_2) - in_2), shape [B, C, H, W]
        scale2_reshaped: scale2 reshaped to [C, 1, 1]
    """
    B, C, H, W = in_2.shape
    
    # Allocate output
    out = torch.empty_like(in_2)
    
    # Launch kernel with autotuning
    num_elements = B * C * H * W
    grid = (num_elements + 512 - 1) // 512  # Use min BLOCK_SIZE for grid
    
    fused_layer_scale_kernel[grid](
        in_ptr=in_2,
        scale1_ptr=scale1,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
    )
    
    # Handle scale2 reshape separately (simple operation - just reshape)
    scale2_reshaped = scale2.reshape(C, 1, 1)
    
    return out, scale2_reshaped


def replacement_func():
    """Return the replacement function."""
    return fused_layer_scale