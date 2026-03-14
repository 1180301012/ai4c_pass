import torch
import triton
import triton.language as tl


# Pattern matching function - matches conv2d + gelu + dropout(0.0)
# The dropout with p=0.0 is a no-op, so we fuse conv + gelu
# This pass targets groups=1024 (depthwise conv with 1024 channels)
def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d (groups=1024) + gelu + dropout(0.0)
    - in_0: bias tensor
    - in_1: weight tensor (for depthwise conv, shape is [1024, 1, 3, 3])
    - in_2: input tensor
    
    Note: dropout with p=0.0 and training=False is a no-op
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 1024)
    tmp_3 = torch.nn.functional.gelu(tmp_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2)


# Optimized Triton kernel for fused conv2d + gelu
# For depthwise convolutions with groups=1024
@triton.jit
def fused_conv2d_gelu_kernel_1024(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, K, H, W, 
    stride_iN, stride_iK, stride_iH, stride_iW,
    stride_wK, stride_wH, stride_wW,
    stride_oN, stride_oK, stride_oH, stride_oW,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for depthwise convolution + GELU
    For depthwise conv with groups=1024
    """
    # Grid: (N, K, H*W / BLOCK_SIZE)
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Calculate spatial offset
    spatial_offset = pid_spatial * BLOCK_SIZE
    spatial_offsets = spatial_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    spatial_mask = spatial_offsets < (H * W)
    
    h = spatial_offsets // W
    w = spatial_offsets % W
    
    # Load bias for this channel
    bias = tl.load(bias_ptr + pid_k)
    
    # Compute depthwise conv - initialize as vector for type consistency
    conv_result = tl.zeros((BLOCK_SIZE,), tl.float32)
    
    # Fully unroll the 3x3 kernel for better type inference
    # Kernel offsets relative to center (for same padding)
    # kh, kw in {-1, 0, 1}
    # h + kh - 1 gives us h-1, h, h+1
    for kh in range(3):
        for kw in range(3):
            h_in = h + kh - 1
            w_in = w + kw - 1
            
            # Bounds check - produce vector mask
            h_valid = (h >= kh) & (h < (H + kh - 1))
            w_valid = (w >= kw) & (w < (W + kw - 1))
            valid = h_valid & w_valid
            
            # Input index - need to compute base offset for each element
            # input[b, c, h, w] = input_ptr + b*stride_iN + c*stride_iK + h*stride_iH + w*stride_iW
            input_offsets = (pid_n * stride_iN + pid_k * stride_iK + 
                            h_in * stride_iH + w_in * stride_iW)
            
            # Weight index - same weight for all spatial positions in same channel
            weight_idx = pid_k * stride_wK + 0 * stride_wH + kh * stride_wW + kw * stride_wW
            
            # Load input values (vector) and weight (scalar)
            input_val = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)
            weight_val = tl.load(weight_ptr + weight_idx)
            
            # Accumulate - now both are vectors
            conv_result = conv_result + input_val * weight_val
    
    # Add bias - need to broadcast scalar to vector
    conv_result = conv_result + bias
    
    # GELU activation using sigmoid
    # tanh(x) = 2 * sigmoid(2*x) - 1
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #         = 0.5 * x * (1 + 2 * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3)) - 1)
    #         = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    
    x = conv_result
    x3 = x * x * x
    sigmoid_arg = 1.702 * (x + coeff * x3)  # 2 * sqrt(2/pi) ≈ 1.702
    gelu_result = x * (1.0 / (1.0 + tl.exp(-sigmoid_arg)))
    
    # Store result
    output_idx = (pid_n * stride_oN + pid_k * stride_oK + 
                  h * stride_oH + w * stride_oW)
    tl.store(output_ptr + output_idx, gelu_result, mask=spatial_mask)


@torch.fx.wrap
def fused_conv2d_gelu_kernel_wrapper_1024(bias, weight, input_tensor):
    """
    Wrapper for fused conv2d (groups=1024) + gelu kernel
    """
    N, C_in, H, W = input_tensor.shape
    C_out, _, kh, kw = weight.shape
    
    # Allocate output
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid calculation
    BLOCK_SIZE = 64
    spatial_elements = H * W
    num_spatial_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (N, C_out, num_spatial_programs)
    
    # Strides
    stride_iN = input_tensor.stride(0)
    stride_iK = input_tensor.stride(1)
    stride_iH = input_tensor.stride(2)
    stride_iW = input_tensor.stride(3)
    
    stride_wK = weight.stride(0)
    stride_wH = weight.stride(2)
    stride_wW = weight.stride(3)
    
    stride_oN = output.stride(0)
    stride_oK = output.stride(1)
    stride_oH = output.stride(2)
    stride_oW = output.stride(3)
    
    fused_conv2d_gelu_kernel_1024[grid](
        input_tensor, weight, bias, output,
        N, C_out, H, W,
        stride_iN, stride_iK, stride_iH, stride_iW,
        stride_wK, stride_wH, stride_wW,
        stride_oN, stride_oK, stride_oH, stride_oW,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def replacement_func():
    """Return the replacement function"""
    return fused_conv2d_gelu_kernel_wrapper_1024