import torch
import triton
import triton.language as tl
import math

# Pattern matching function - exactly matches the computation structure
def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for Conv2D + HardTanh + Element-wise multiply fusion.
    This matches: Conv2D(in_2, in_1, in_0) -> HardTanh(in_3) -> Multiply
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments from matched pattern nodes for the fused kernel.
    """
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused Conv2D + HardTanh + Multiply
@triton.jit
def fused_conv2d_hardtanh_multiply_kernel(
    # Output tensor pointers
    out_ptr,
    # Input tensor pointers
    input_ptr,      # in_2: [N, C_in, H, W]
    hardtanh_input_ptr,  # in_3: [N, C_out, H, W]
    # Weight tensor pointers
    weight_ptr,     # in_1: [C_out, C_in, KH, KW]
    bias_ptr,       # in_0: [C_out]
    # Tensor shapes and strides
    N, C_in, H, W, C_out,
    input_stride_N, input_stride_C_in, input_stride_H, input_stride_W,
    hardtanh_stride_N, hardtanh_stride_C_out, hardtanh_stride_H, hardtanh_stride_W,
    weight_stride_C_out, weight_stride_C_in, weight_stride_KH, weight_stride_KW,
    bias_stride_C_out,
    output_stride_N, output_stride_C_out, output_stride_H, output_stride_W,
    # Hardtanh parameters
    hardtanh_min, hardtanh_max,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Number of batches to process per program
    BLOCK_SIZE_N: tl.constexpr,  # Number of channels to process per program  
    BLOCK_SIZE_HW: tl.constexpr  # Number of height+width positions to process per program
):
    # Get program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    hw_start = pid_hw * BLOCK_SIZE_HW
    m_end = min(m_start + BLOCK_SIZE_M, N)
    n_end = min(n_start + BLOCK_SIZE_N, C_out)
    hw_end = min(hw_start + BLOCK_SIZE_HW, H * W)
    
    # Early exit if out of bounds
    if m_start >= N:
        return
    if n_start >= C_out:
        return
    if hw_start >= H * W:
        return
    
    # Process each height-width position in block
    for hw_idx in range(hw_start, hw_end):
        # Convert flat index to 2D coordinates
        h = hw_idx // W
        w = hw_idx % W
        # Skip if out of bounds
        valid_position = (h < H) and (w < W)
        if valid_position:
            # Use fixed-size channel range with proper masking
            actual_channels = min(n_end - n_start, BLOCK_SIZE_N)
            channel_range = tl.arange(0, BLOCK_SIZE_N)  # Fixed size compile-time constant
            channel_mask = channel_range < actual_channels
            
            # Calculate global channel indices
            global_channel_range = n_start + channel_range
            
            # Load hardtanh input for this position
            hardtanh_ptrs = (
                hardtanh_input_ptr + 
                m_start * hardtanh_stride_N + 
                (global_channel_range * hardtanh_stride_C_out) +
                h * hardtanh_stride_H + 
                w * hardtanh_stride_W
            )
            hardtanh_vals = tl.load(hardtanh_ptrs, mask=channel_mask, other=0.0)
            
            # Apply hardtanh
            hardtanh_vals = tl.where(hardtanh_vals < hardtanh_min, hardtanh_min, hardtanh_vals)
            hardtanh_vals = tl.where(hardtanh_vals > hardtanh_max, hardtanh_max, hardtanh_vals)
            
            # Load bias for this channel range
            bias_ptrs = bias_ptr + global_channel_range
            bias_vals = tl.load(bias_ptrs, mask=channel_mask, other=0.0)
            
            # Process convolution
            conv_vals = bias_vals  # [BLOCK_SIZE_N]
            for c_in in range(C_in):
                # Load weights for this output channel and input channel
                weight_ptrs = weight_ptr + (global_channel_range * weight_stride_C_out) + c_in * weight_stride_C_in
                weight_vals = tl.load(weight_ptrs, mask=channel_mask, other=0.0)
                
                # Load input data for this input channel and spatial position
                input_ptr_val = input_ptr + m_start * input_stride_N + c_in * input_stride_C_in + h * input_stride_H + w
                input_vals = tl.load(input_ptr_val, mask=True, other=0.0)
                
                # Accumulate convolution
                conv_vals += weight_vals * input_vals
            
            # Apply hardtanh and multiply
            result = hardtanh_vals * conv_vals
            
            # Store output
            out_ptrs = (
                out_ptr + 
                m_start * output_stride_N + 
                (global_channel_range * output_stride_C_out) +
                h * output_stride_H + 
                w * output_stride_W
            )
            tl.store(out_ptrs, result, mask=channel_mask)

# Kernel wrapper function
@torch.fx.wrap
def fused_conv2d_hardtanh_multiply(bias, weight, input_tensor, hardtanh_input):
    """
    Fused kernel combining Conv2D + HardTanh + Element-wise multiplication.
    """
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get tensor shapes and strides
    N, C_in, H, W = input_tensor.shape
    K = weight.shape[0]  # C_out
    KH, KW = weight.shape[2], weight.shape[3]  # Should be 1x1
    
    # Validate shapes
    assert K == bias.shape[0], "Output channels mismatch"
    assert hardtanh_input.shape == (N, K, H, W), "Hardtanh input shape mismatch"
    assert KH == 1 and KW == 1, "Only 1x1 convolution supported"
    
    # Calculate strides
    input_stride = input_tensor.stride()
    hardtanh_stride = hardtanh_input.stride()
    weight_stride = weight.stride()
    bias_stride = bias.stride()
    output_shape = (N, K, H, W)
    out = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    output_stride = out.stride()
    
    # Block sizes configuration
    BLOCK_SIZE_M = 2   # Number of batches to process together
    BLOCK_SIZE_N = 64  # Number of output channels to process per program
    BLOCK_SIZE_HW = 64  # Number of height+width positions to process per program
    
    # Calculate grid dimensions
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (K + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch kernel with 3D grid
    fused_conv2d_hardtanh_multiply_kernel[grid_m, grid_n, grid_hw](
        out,                                # output
        input_tensor,                      # input tensor
        hardtanh_input,                    # hardtanh input
        weight,                            # weight tensor
        bias,                              # bias tensor
        N, C_in, H, W, K,
        input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        hardtanh_stride[0], hardtanh_stride[1], hardtanh_stride[2], hardtanh_stride[3],
        weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
        bias_stride[0],
        output_stride[0], output_stride[1], output_stride[2], output_stride[3],
        0.0, 6.0,  # HardTanh parameters: min=0.0, max=6.0
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_HW
    )
    
    return out

# Replacement function returns the kernel wrapper
def replacement_func():
    return fused_conv2d_hardtanh_multiply