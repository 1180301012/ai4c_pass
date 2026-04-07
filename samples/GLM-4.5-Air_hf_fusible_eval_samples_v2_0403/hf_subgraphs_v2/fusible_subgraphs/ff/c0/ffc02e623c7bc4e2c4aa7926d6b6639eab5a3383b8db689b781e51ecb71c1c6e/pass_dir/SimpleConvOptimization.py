import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor):
    """Match convolution with specific parameters"""
    result = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (1, 1), (1, 1), 1)
    return result

def replacement_args(input_tensor, weight_tensor):
    return (input_tensor, weight_tensor)

@triton.jit
def simple_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program handles (BLOCK_SIZE_M * BLOCK_SIZE_N) output elements
    m = (pid // BLOCK_SIZE_N) * BLOCK_SIZE_M
    n = (pid % BLOCK_SIZE_N) * BLOCK_SIZE_N
    
    # Create masks
    mask_m = m < batch_size * out_channels
    mask_n = n < height * width
    
    # Process a tile of output
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if (m + i) < batch_size * out_channels and (n + j) < height * width:
                # Calculate indices
                batch_idx = (m + i) // out_channels
                out_channel_idx = (m + i) % out_channels
                
                spatial_idx = n + j
                h_idx = spatial_idx // width
                w_idx = spatial_idx % width
                
                # Output position
                output_pos = (m + i) * height * width + (n + j)
                
                # Initialize convolution result
                acc = 0.0
                for oc in range(out_channels):
                    for ic in range(in_channels):
                        for kh in range(3):  # 3x3 kernel
                            for kw in range(3):
                                # Calculate input position with padding
                                h_in = h_idx + kh - 1  # padding of 1
                                w_in = w_idx + kw - 1
                                
                                # Only process valid positions (within bounds)
                                if 0 <= h_in < height and 0 <= w_in < width:
                                    input_pos = batch_idx * height * width * in_channels + \
                                               h_in * width * in_channels + \
                                               w_in * in_channels + ic
                                    weight_pos = oc * in_channels * 3 * 3 + \
                                               ic * 3 * 3 + \
                                               kh * 3 + kw
                                    
                                    input_val = tl.load(input_ptr + input_pos, other=0.0)
                                    weight_val = tl.load(weight_ptr + weight_pos, other=0.0)
                                    acc += input_val * weight_val
                
                tl.store(output_ptr + output_pos, acc)

@torch.fx.wrap
def optimized_conv(input_tensor, weight_tensor):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, kernel_h, kernel_w, _ = weight_tensor.shape
    
    # Validate kernel size
    assert kernel_h == 3 and kernel_w == 3, "This kernel expects 3x3 convolution"
    
    # Output shape for padding 1, stride 1 is same as input
    output_shape = (batch_size, out_channels, height, width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use smaller block sizes to avoid tensor size limits
    BLOCK_SIZE_M = 8   # Output channels per thread group
    BLOCK_SIZE_N = 8   # Spatial positions per thread group
    
    # Grid calculation
    total_elements = batch_size * out_channels * height * width
    grid_size = (total_elements + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    input_contiguous = input_tensor.contiguous()
    weight_contiguous = weight_tensor.contiguous()
    
    simple_conv_kernel[(grid_size,)](
        input_contiguous,
        weight_contiguous,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return optimized_conv