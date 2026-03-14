import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matching: Conv2D (1x1) → AvgPool2d (2x2) fusion"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_0 = None
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    tmp_1 = None
    return (tmp_2,)

def replacement_args(in_0, in_1):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1)

@triton.jit
def fused_conv_avgpool_kernel(
    input_ptr,          # Input tensor pointer [N, C_in, H, W]
    weight_ptr,         # Weight tensor [C_out, C_in, 1, 1]
    output_ptr,         # Output tensor [N, C_out, H_out, W_out]
    N, C_in, H, W,      # Input dimensions
    C_out,              # Output channels (from weight shape[0])
    H_out, W_out,       # Output dimensions after pooling
    BLOCK_SIZE_C: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_HW: tl.constexpr,  # Block size for spatial dimensions (combined height+width)
):
    """Fused Conv2D (1x1) + AvgPool2d (2x2) kernel"""
    
    # Get program IDs (3D grid)
    out_idx = tl.program_id(0)  # Output position (flattened)
    batch_idx = tl.program_id(1)  # Batch index
    pad_idx = tl.program_id(2)    # Padding for extra parallelism
    
    # Decode output position into channel and coordinates
    # Flatten H_out * W_out coordinates
    total_spatial = H_out * W_out
    c_out_idx = out_idx // total_spatial
    hw_idx = out_idx % total_spatial
    h_pool_idx = hw_idx // W_out
    w_pool_idx = hw_idx % W_out
    
    # Only process if within bounds
    mask = (c_out_idx < C_out) & (batch_idx < N) & (h_pool_idx < H_out) & (w_pool_idx < W_out)
    if not mask:
        return
    
    # Map output pooling coordinates to input coordinates
    h_in_start = h_pool_idx * 2
    w_in_start = w_pool_idx * 2
    
    # Compute 1x1 convolution at each position in 2x2 window
    conv_sum = 0.0
    valid_positions = 0
    
    for dh in range(2):
        for dw in range(2):
            h_in = h_in_start + dh
            w_in = w_in_start + dw
            
            if (h_in < H) and (w_in < W):  # Only valid positions
                # Compute 1x1 convolution at this position
                conv_val = 0.0
                for c_in_idx in range(C_in):
                    # Load weight for this input/output channel pair
                    weight_idx = c_out_idx * C_in + c_in_idx
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Load input value with bounds check
                    input_idx = (
                        batch_idx * (C_in * H * W) +
                        c_in_idx * (H * W) +
                        h_in * W +
                        w_in
                    )
                    input_val = tl.load(input_ptr + input_idx)
                    
                    conv_val += weight_val * input_val
                
                conv_sum += conv_val
                valid_positions += 1
    
    # Average pooling
    if valid_positions > 0:
        avg_val = conv_sum / valid_positions
    else:
        avg_val = 0.0
    
    # Store result to output
    output_idx = (
        c_out_idx * (N * H_out * W_out) +
        batch_idx * (H_out * W_out) +
        h_pool_idx * W_out +
        w_pool_idx
    )
    tl.store(output_ptr + output_idx, avg_val)
    
@torch.fx.wrap
def fused_conv_avgpool(in_0, in_1):
    """Wrapper function for the fused Conv2D + AvgPool2d operation"""
    # Get input shapes
    N, C_in, H, W = in_1.shape
    C_out, _, _, _ = in_0.shape
    
    # Calculate output dimensions after pooling
    H_out = H // 2
    W_out = W // 2
    
    # Create output tensor
    output = torch.empty(N, C_out, H_out, W_out, dtype=in_1.dtype, device=in_1.device)
    
    # Set block sizes based on typical GPU architecture
    BLOCK_SIZE_C = 256      # Output channels per block
    BLOCK_SIZE_HW = 1       # Process one spatial position at a time (simpler)
    
    # Calculate grid dimensions (using 3D grid)
    # Flatten all output positions (channels * spatial)
    total_output_elements = C_out * H_out * W_out
    grid_0 = max(1, (total_output_elements + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C)
    grid_1 = max(1, (N + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)  # Batch dimension
    grid_2 = 8              # Third dimension for extra parallelism
    
    # Launch kernel with 3D grid
    fused_conv_avgpool_kernel[grid_0, grid_1, grid_2](
        input_ptr=in_1,
        weight_ptr=in_0,
        output_ptr=output,
        N=N, C_in=C_in, H=H, W=W,
        C_out=C_out, H_out=H_out, W_out=W_out,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_avgpool