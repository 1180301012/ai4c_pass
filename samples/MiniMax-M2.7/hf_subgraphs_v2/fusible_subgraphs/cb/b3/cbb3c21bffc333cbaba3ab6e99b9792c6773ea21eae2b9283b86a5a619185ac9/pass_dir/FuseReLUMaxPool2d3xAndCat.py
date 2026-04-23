import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0):
    """
    Match the pattern: relu -> 3x max_pool2d (same params, same input) -> cat
    The key insight is all 3 max_pool2d operations produce identical results.
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def relu_maxpool2d_cat_kernel(
    input_ptr,
    output_ptr,
    stride_n, stride_c, stride_h, stride_w,
    N, C, H, W,
):
    """
    Fused kernel: relu -> max_pool2d(3x) -> cat
    Each program handles one output element.
    Output has 4x channels: [N, 4*C, H, W]
    """
    pid = tl.program_id(0)
    
    # Calculate output position indices
    n = pid // (4 * C * H * W)
    remainder = pid % (4 * C * H * W)
    out_c = remainder // (C * H * W)
    remainder2 = remainder % (C * H * W)
    h = remainder2 // (H * W)
    w = remainder2 % (H * W)
    
    # Input channel (cyclic for the 4 output channel groups)
    in_c = out_c % C
    
    # Load input and apply ReLU
    input_offset = n * stride_n + in_c * stride_c + h * stride_h + w * stride_w
    val = tl.load(input_ptr + input_offset)
    relu_val = tl.where(val > 0, val, 0.0)
    
    # Compute max pooling over 5x5 window (kernel=5, padding=2, stride=1)
    max_pool_val = relu_val
    for kh in range(-2, 3):
        for kw in range(-2, 3):
            ih = h + kh
            iw = w + kw
            # Check bounds (with padding=2, valid indices are 0..H-1, 0..W-1)
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                offset = n * stride_n + in_c * stride_c + ih * stride_h + iw * stride_w
                v = tl.load(input_ptr + offset)
                v = tl.where(v > 0, v, 0.0)  # ReLU on each neighbor
                max_pool_val = tl.maximum(max_pool_val, v)
    
    # Select based on output channel group:
    # Group 0 (c=0..C-1): relu result
    # Groups 1,2,3 (c=C..4*C-1): max_pool result (identical tiles)
    result = tl.where(out_c < C, relu_val, max_pool_val)
    
    # Store at output position
    output_offset = pid
    tl.store(output_ptr + output_offset, result)


@torch.fx.wrap
def triton_relu_maxpool2d_cat(x):
    """Fused ReLU + MaxPool2d(3x) + Cat optimization"""
    # Input shape: [N, C, H, W]
    # Output shape: [N, 4*C, H, W] - 4x channels (relu + 3x maxpool tiled)
    
    N, C, H, W = x.shape
    out_channels = C * 4
    n_elements = N * out_channels * H * W
    
    # Allocate output tensor
    out = torch.empty((N, out_channels, H, W), device=x.device, dtype=x.dtype)
    
    # Launch kernel with 1D grid (one program per output element)
    grid = (n_elements,)
    
    relu_maxpool2d_cat_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        N, C, H, W,
    )
    
    return out


def replacement_func():
    return triton_relu_maxpool2d_cat