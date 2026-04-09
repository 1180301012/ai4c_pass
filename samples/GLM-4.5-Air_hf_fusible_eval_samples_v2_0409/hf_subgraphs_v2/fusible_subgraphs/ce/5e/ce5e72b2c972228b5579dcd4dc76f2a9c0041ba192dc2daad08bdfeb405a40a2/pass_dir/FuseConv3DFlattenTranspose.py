import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv3d_flatten_transpose_kernel(
    input_ptr,      # Input tensor [B, C, D, H, W]
    weight_ptr,     # Weight tensor [O, C, kD, kH, kW]
    bias_ptr,       # Bias tensor [O]
    output_ptr,     # Output tensor [B, O, HW, D]
    B, C, D, H, W,  # Input dimensions
    O, kD, kH, kW,  # Weight dimensions
    output_HD,      # Output HW*D dimension
    stride0, stride1, stride2, stride3, stride4,  # Input strides
    weight_stride0, weight_stride1, weight_stride2, weight_stride3, weight_stride4,
    bias_stride0,
    output_stride0, output_stride1, output_stride2, output_stride3,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused Conv3D + Flatten + Transpose kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine which output elements we're responsible for
    # Output tensor is [B, O, HW, D]
    batch_id = pid_m // output_HD
    elem_id = pid_m % output_HD  # Linear position in HW*D flattened space
    
    # Unpack HW*D into H_pos, W_pos, D_pos
    D_pos = elem_id // (H * W)
    remaining = elem_id % (H * W)
    H_pos = remaining // W
    W_pos = remaining % W
    
    feature_id = pid_n  # Output feature channel
    
    # Only process valid indices
    if batch_id >= B or feature_id >= O or D_pos >= D or H_pos >= H or W_pos >= W:
        return
    
    # Calculate output position in flattened HW*D space
    output_hw_d_pos = H_pos * W + W_pos + D_pos * H * W
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform convolution
    for c in range(C):  # Input channels
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input indices with padding=0, stride=1
                    in_d = D_pos + kd - (kD - 1) // 2
                    in_h = H_pos + kh - (kH - 1) // 2
                    in_w = W_pos + kw - (kW - 1) // 2
                    
                    # Only process if within input bounds (no padding)
                    if in_d >= 0 and in_d < D and in_h >= 0 and in_h < H and in_w >= 0 and in_w < W:
                        # Load input value
                        input_offset = batch_id * stride0 + c * stride1 + in_d * stride2 + in_h * stride3 + in_w * stride4
                        input_val = tl.load(input_ptr + input_offset)
                        
                        # Load weight value
                        weight_offset = feature_id * weight_stride0 + c * weight_stride1 + kd * weight_stride2 + kh * weight_stride3 + kw * weight_stride4
                        weight_val = tl.load(weight_ptr + weight_offset)
                        
                        acc += input_val * weight_val
    
    # Add bias
    bias_offset = feature_id * bias_stride0
    bias_val = tl.load(bias_ptr + bias_offset)
    acc += bias_val
    
    # Store result at [batch_id, feature_id, output_hw_d_pos]
    output_offset = batch_id * output_stride0 + feature_id * output_stride1 + output_hw_d_pos * output_stride2
    tl.store(output_ptr + output_offset, acc)

def pattern(in_3, in_1, in_0):
    """Matches Conv3D + Flatten + Transpose sequence"""
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

def replacement_args(in_3, in_1, in_0):
    """Extract the arguments needed for the fused operation"""
    return (in_3, in_1, in_0)

@torch.fx.wrap
def fused_conv3d_flatten_transpose(input_tensor, weight_tensor, bias_tensor):
    """Fused operation: Conv3D + Flatten + Transpose"""
    # Get input dimensions
    B, C, D, H, W = input_tensor.shape
    O, C_k, kD, kH, kW = weight_tensor.shape
    
    # For this specific pattern, we know the convolution parameters
    output_depth = D - kD + 1  # No padding, stride=1
    output_height = H - kH + 1
    output_width = W - kW + 1
    output_HW = output_height * output_width  # This should be 1 based on the original pattern
    
    # Output tensor shape: [B, O, HW*D] -> [B, O, 1*D] since HW=1
    output_shape = (B, O, output_HW * output_depth)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # For simplicity, use a basic block size (this could be optimized further)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Calculate grid dimensions
    total_elements = B * output_HW * output_depth
    grid_m = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (O + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Get strides (simplified - assuming contiguous tensors)
    input_stride = [1, 1, 1, 1, 1]
    weight_stride = [1, 1, 1, 1, 1]
    bias_stride = [1]
    output_stride = [1, 1, 1, 1]  # [B, O, HW*D]
    
    # Launch kernel
    fused_conv3d_flatten_transpose_kernel[grid_m, grid_n](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        B, C, D, H, W,
        O, kD, kH, kW,
        output_HW * output_depth,
        input_stride[0], input_stride[1], input_stride[2], input_stride[3], input_stride[4],
        weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3], weight_stride[4],
        bias_stride[0],
        output_stride[0], output_stride[1], output_stride[2], output_stride[3],
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    """Returns the fused kernel function"""
    return fused_conv3d_flatten_transpose