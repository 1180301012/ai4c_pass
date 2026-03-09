import torch
import triton
import triton.language as tl

def pattern(in_7, tmp_5, tmp_4, in_6):
    # Conv2D followed by Sigmoid followed by Element-wise multiplication
    # This matches the common pattern: conv -> sigmoid -> mul
    tmp_6 = torch.conv2d(in_7, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
    tmp_7 = tmp_6.sigmoid()
    tmp_8 = in_6 * tmp_7
    return tmp_8

def replacement_args(in_7, tmp_5, tmp_4, in_6):
    return (in_7, tmp_5, tmp_4, in_6)

@triton.jit
def fused_conv_sigmoid_mul_kernel(
    input_ptr,      # in_7: [N, C_in, 1, 1] 
    weight_ptr,     # tmp_5: [C_out, C_in/groups, 1, 1]
    bias_ptr,       # tmp_4: [C_out]
    mul_ptr,        # in_6: [N, C_in, H, W]
    output_ptr,     # output: [N, C_out, H, W]
    N, C_out, C_in, H, W,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial and output channel position
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1) 
    pid_output_idx = tl.program_id(2)
    
    # Flatten N and C_out dimensions 
    total_elements = N * C_out
    if pid_output_idx >= total_elements:
        return
        
    # Unflatten the output index
    pid_n = pid_output_idx // C_out
    pid_c = pid_output_idx % C_out
    
    # Calculate output position
    output_h = pid_h * stride_h - pad_h
    output_w = pid_w * stride_w - pad_w
    
    # Check bounds for output position
    if (output_h >= H) or (output_w >= W) or (output_h < 0) or (output_w < 0):
        return
    
    # Calculate weight pointer offset for this output channel
    weight_offset = pid_c * (C_in // groups) * 1 * 1
    bias_offset = pid_c
    
    # Load bias
    bias = tl.load(bias_ptr + bias_offset, other=0.0)
    
    # Initialize output sum
    output_sum = bias
    
    # Process each input channel group
    for c_in_group in range(0, C_in // groups, BLOCK_SIZE):
        offsets = c_in_group + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (C_in // groups)
        
        # Load weights
        weight_ptrs = weight_ptr + weight_offset + offsets * 1 * 1
        weights = tl.load(weight_ptrs, mask=mask, other=0.0)
        
        # Load input from spatial location (1,1)
        input_ptrs = input_ptr + (
            pid_n * C_in * 1 * 1 + 
            (c_in_group + offsets) * 1 * 1
        )
        inputs = tl.load(input_ptrs, mask=mask, other=0.0)
        
        # Matrix multiply contribution
        output_sum += tl.sum(weights * inputs)
    
    # Apply sigmoid
    output_val = 1.0 / (1.0 + tl.exp(-output_sum))
    
    # Store to all spatial positions for this batch and output channel
    for h in range(H):
        for w in range(W):
            output_ptrs = output_ptr + (
                pid_n * C_out * H * W + pid_c * H * W + h * W + w
            )
            tl.store(output_ptrs, output_val)

@torch.fx.wrap
def fused_conv_sigmoid_mul(in_7, tmp_5, tmp_4, in_6):
    # Get input shapes
    N, C_in_conv, H_in_conv, W_in_conv = in_7.shape
    C_out_conv, C_in_weight, kh, kw = tmp_5.shape
    N_mul, C_in_mul, H_mul, W_mul = in_6.shape
    
    # Assert shapes match the expected patterns
    assert kh == 1 and kw == 1, "Only 1x1 convolutions supported"
    assert C_in_conv == C_in_weight, "Input channel mismatch between conv input and weight"
    assert C_out_conv == C_in_mul, "Output channels must match multiplication tensor"
    assert H_in_conv == 1 and W_in_conv == 1, "SE input must be 1x1"
    
    # Output shape matches in_6 shape
    output_shape = in_6.shape
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=in_6.dtype, device=in_6.device)
    
    # Kernel launch parameters
    BLOCK_SIZE = 32  # Number of input channels to process per iteration
    
    # Grid size: (H_out, W_out, N*C_out) - 3D grid as required by Triton
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 0, 0
    dilation_h, dilation_w = 1, 1
    groups = 1
    
    # Calculate grid dimensions
    H_out, W_out = H_mul, W_mul
    total_output_elements = N * C_out_conv
    
    grid = (
        (H_out + 31) // 32,                     # H blocks
        (W_out + 31) // 32,                     # W blocks  
        (total_output_elements + 31) // 32,     # Combined N*C_out blocks
    )
    
    # Launch kernel
    fused_conv_sigmoid_mul_kernel[grid](
        in_7,
        tmp_5, 
        tmp_4,
        in_6,
        output,
        N, C_out_conv, C_in_conv, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv_sigmoid_mul