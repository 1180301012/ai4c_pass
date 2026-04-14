import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Pattern matches conv2d with stride (1,1) followed by mean over spatial dimensions (2, 3)
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv2d_mean_kernel(
    input_ptr,      # Input tensor [N, C_in, H, W]
    weight_ptr,     # Weight tensor [C_out, kH, kW, C_in//groups]
    conv_out_ptr,   # Full convolution output buffer [N, C_out, H_out, W_out]
    mean_out_ptr,   # Mean output buffer [N, C_out, 1, 1]
    N, C_in, H, W, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Map program to output elements
    pid_n = tl.program_id(0)  # Batch dimension
    pid_c = tl.program_id(1)  # Output channel dimension
    
    # Initialize summed values for mean computation
    spatial_sum = 0.0
    conv_count = 0
    
    # Process spatial positions using grid within the program
    pid_h = tl.program_id(2)  # Grid dimension for height
    pid_w = tl.program_id(3)  # Grid dimension for width
    
    # Calculate spatial position for this thread
    h = pid_h * BLOCK_SIZE_HW
    w = pid_w * BLOCK_SIZE_HW
    
    # Process a block of spatial positions
    for dh in range(BLOCK_SIZE_HW):
        for dw in range(BLOCK_SIZE_HW):
            h_idx = h + dh
            w_idx = w + dw
            
            if h_idx < H_out and w_idx < W_out:
                # Calculate input position with padding and stride
                ih = h_idx * stride_h - pad_h
                iw = w_idx * stride_w - pad_w
                
                # Initialize convolution for this spatial position
                conv_val = 0.0
                
                # Perform 3x3 convolution if input position is valid
                if ih >= 0 and iw >= 0 and ih + 2 < H and iw + 2 < W:
                    # Get weight index for this output channel
                    weight_start = pid_c * 3 * 3
                    
                    # Load and apply 3x3 weights
                    for kh in range(3):
                        for kw in range(3):
                            # Weight index
                            wi = weight_start + kh * 3 + kw
                            
                            # Input position with dilation
                            input_h = ih + kh * dilation_h
                            input_w = iw + kw * dilation_w
                            
                            # Input offset
                            input_offset = (pid_n * C_in * H * W + 
                                          input_h * W + input_w)
                            
                            # Load input and weight
                            input_val = tl.load(input_ptr + input_offset).to(tl.float32)
                            weight_val = tl.load(weight_ptr + wi).to(tl.float32)
                            
                            # Apply convolution
                            conv_val += input_val * weight_val
                            
                            # Accumulate for mean computation
                            spatial_sum += conv_val
                            conv_count += 1
                
                # Store convolution result
                output_offset = (pid_n * C_out * H_out * W_out + 
                               pid_c * H_out * W_out + h_idx * W_out + w_idx)
                tl.store(conv_out_ptr + output_offset, conv_val.to(input_ptr.type.element_ty))
    
    # Compute and store mean
    if conv_count > 0:
        mean_val = spatial_sum / conv_count
        mean_offset = pid_n * C_out + pid_c
        tl.store(mean_out_ptr + mean_offset, mean_val.to(input_ptr.type.element_ty))

@torch.fx.wrap
def fused_conv2d_mean(in_0, in_1):
    # Get input shapes
    N, C_in, H, W = in_1.shape
    C_out = in_0.shape[0]
    
    # Parameters for conv2d with stride (1,1)
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    groups = 384
    
    # Compute output dimensions
    H_out = (H + 2*pad_h - 3*dilation_h) // stride_h + 1
    W_out = (W + 2*pad_w - 3*dilation_w) // stride_w + 1
    
    # Create output tensors with correct shapes
    conv_out = torch.empty(N, C_out, H_out, W_out, dtype=in_1.dtype, device=in_1.device)
    mean_out = torch.empty(N, C_out, 1, 1, dtype=in_1.dtype, device=in_1.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = min(64, C_out)
    BLOCK_SIZE_HW = 4  # Smaller tile for spatial dimensions
    
    # Calculate grid dimensions - now 4D grid for batch, channels, height, width
    grid_n = N
    grid_c = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H_out + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid_w = (W_out + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    grid = (grid_n, grid_c, grid_h, grid_w)
    
    # Launch kernel
    fused_conv2d_mean_kernel[grid](
        in_1, in_0, conv_out, mean_out,
        N, C_in, H, W, H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    
    return conv_out, mean_out

def replacement_func():
    return fused_conv2d_mean