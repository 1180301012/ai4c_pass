import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching function for Conv2D + AvgPool2D fusion
    Matches: Conv2D (1x1) followed by AvgPool2D (2x2, stride 2)
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2

def replacement_args(in_0, in_1):
    """
    Extract arguments for the fused kernel
    """
    return (in_0, in_1)

@triton.jit
def fused_conv_avgpool_kernel(
    input_ptr,      # Input feature map [N, C_in, H, W]
    weight_ptr,     # Conv weights [C_out, C_in, 1, 1]
    output_ptr,     # Output [N, C_out, H_out, W_out]
    N, C_in, H, W,  # Input dimensions
    C_out,          # Output channels
    H_out, W_out,   # Output spatial dimensions
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Fused Conv2D + AvgPool2D kernel
    Performs 1x1 convolution followed by 2x2 average pooling with stride 2
    """
    # Program identifiers
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output coordinates (after pooling)
    h_out = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    c_out = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    h_out_mask = h_out < H_out
    w_out_mask = w_out < W_out
    c_out_mask = c_out < C_out
    n_mask = n < N
    
    # Calculate input coordinates (before pooling: H_in = 2*H_out, W_in = 2*W_out)
    h_in = h_out * 2  # stride 2
    w_in = w_out * 2  # stride 2
    
    # Initialize output accumulator for average pooling
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    count = 0
    
    # Average pooling: sum over 2x2 window
    for dh in range(2):
        for dw in range(2):
            h_in_offset = h_in + dh
            w_in_offset = w_in + dw
            
            # Check if input coordinates are valid
            h_in_mask = h_in_offset < H
            w_in_mask = w_in_offset < W
            
            # Expand masks for broadcasting
            mask = n_mask[:, None, None, None] & c_out_mask[None, :, None, None] & h_in_mask[None, None, :, None] & w_in_mask[None, None, None, :]
            
            # Load input data and weights
            input_offset = (
                n[:, None, None, None] * (C_in * H * W) +
                c_out_mask[None, :, None, None] * (H * W) +  # Use same channels for all input channels
                h_in_offset[None, None, :, None] * W +
                w_in_offset[None, None, None, :]
            )
            input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
            
            # Load weights (1x1 convolution)
            weight_offset = (
                c_out_mask[:, None] * (C_in * 1 * 1) +
                tl.arange(0, BLOCK_SIZE_C)[:, None] * (1 * 1) +  # Input channel offset
                0 * 1 +  # Height offset (1x1 kernel)
                0        # Width offset (1x1 kernel)
            )
            weight_val = tl.load(weight_ptr + weight_offset, mask=c_out_mask[:, None], other=0.0)
            
            # Perform 1x1 convolution (pointwise multiply and sum over input channels)
            conv_val = input_val * weight_val
            conv_val_sum = tl.sum(conv_val, axis=1)  # Sum over input channels
            
            # Accumulate for average pooling
            acc += conv_val_sum
            count += 1
    
    # Average pooling: divide by count (4 for 2x2 window)
    acc = acc / count
    
    # Store output
    output_offset = (
        n[:, None, None, None] * (C_out * H_out * W_out) +
        c_out_mask[None, :, None, None] * (H_out * W_out) +
        h_out[None, None, :, None] * W_out +
        w_out[None, None, None, :]
    )
    tl.store(output_ptr + output_offset, acc, mask=n_mask[:, None, None, None] & c_out_mask[None, :, None, None] & h_out_mask[None, None, :, None] & w_out_mask[None, None, None, :])

@torch.fx.wrap
def fused_conv_avgpool(in_0, in_1):
    """
    Wrapper function for the fused Conv2D + AvgPool2D operation
    """
    # Get tensor shapes
    N, C_in, H, W = in_1.shape
    C_out, _, _, _ = in_0.shape
    H_out, W_out = H // 2, W // 2  # Average pooling with stride 2
    
    # Output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=in_0.dtype, device=in_0.device)
    
    # Determine block sizes based on tensor dimensions
    BLOCK_SIZE_N = min(4, N)
    BLOCK_SIZE_C = min(32, C_out)
    BLOCK_SIZE_H = min(8, H_out)
    BLOCK_SIZE_W = min(8, W_out)
    
    # Calculate grid sizes
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    fused_conv_avgpool_kernel[(grid_n, grid_c, grid_h, grid_w)](
        input_ptr=in_1,
        weight_ptr=in_0,
        output_ptr=out,
        N=N, C_in=C_in, H=H, W=W,
        C_out=C_out,
        H_out=H_out, W_out=W_out,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out

def replacement_func():
    """
    Returns the fused kernel function
    """
    return fused_conv_avgpool