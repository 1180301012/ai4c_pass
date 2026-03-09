import torch
import triton
import triton.language as tl

def pattern(in_5, tmp_3, tmp_2):
    # Match torch conv2d with specific parameters exactly as in model
    return torch.conv2d(in_5, tmp_3, tmp_2, (2, 2), (1, 1), (1, 1), 1)

def replacement_args(in_5, tmp_3, tmp_2):
    return (in_5, tmp_3, tmp_2)

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels, kernel_height, kernel_width, 
    output_height, output_width,
    stride_height, stride_width, pad_height, pad_width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute output coordinates
    output_coords = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = output_coords < (output_height * output_channels)
    
    # Reshape coordinates to be per-output channel
    pid_oh = output_coords // output_channels
    pid_oc = output_coords % output_channels
    
    # Initialize output
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Convolution loop
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for ic in range(input_channels):
                # Compute input coordinates with padding
                ih = pid_oh * stride_height - pad_height + kh
                iw = output_coords // output_channels * stride_width - pad_width + kw
                
                # Bounds checking
                ih_mask = (ih >= 0) & (ih < input_height)
                iw_mask = (iw >= 0) & (iw < input_width)
                
                # Calculate input pointer
                input_ptrs = ih * input_width + iw + ic * input_height * input_width
                
                # Calculate weight pointer
                weight_ptrs = pid_oc * input_channels * kernel_height * kernel_width + ic * kernel_height * kernel_width + kh * kernel_width + kw
                
                # Load data
                input_vals = tl.load(input_ptr + input_ptrs, mask=ih_mask & iw_mask, other=0.0)
                weight_vals = tl.load(weight_ptr + weight_ptrs, other=0.0)
                
                # Accumulate
                accumulator += input_vals * weight_vals
    
    # Add bias
    bias_ptrs = pid_oc
    bias_vals = tl.load(bias_ptr + bias_ptrs, mask=pid_oc < output_channels, other=0.0)
    accumulator += bias_vals
    
    # Store result
    output_ptrs = output_coords
    tl.store(output_ptr + output_ptrs, accumulator, mask=mask)

@torch.fx.wrap
def optimized_conv2d(input, weight, bias):
    # Get input dimensions
    B, C_in, H_in, W_in = input.shape
    C_out, _, K_H, K_W = weight.shape
    
    # Calculate output dimensions with padding and stride
    H_out = (H_in + 2 * 1 - K_H) // 2 + 1  # (48 + 2 - 3) // 2 + 1 = 24
    W_out = (W_in + 2 * 1 - K_W) // 2 + 1  # (48 + 2 - 3) // 2 + 1 = 24
    
    # Reshape input to [B, C_in, H_out*W_out] for simpler kernel
    input_flat = input.reshape(B, C_in, -1)  # [1, 192, 48*48]
    
    # Calculate total output elements
    total_output = B * C_out * H_out * W_out
    output_per_program = 384 * 24 * 24  # 1 * 384 * 24 * 24 = 221184
    
    # Create output tensor (flattened first, then reshape)
    output_flat = torch.empty((B, C_out, H_out * W_out), dtype=input.dtype, device=input.device)
    
    # Triton kernel configuration
    BLOCK_SIZE_M = 1024  # Number of output elements per program
    
    # Calculate grid size
    grid_size = triton.cdiv(total_output, BLOCK_SIZE_M)
    
    # Launch kernel - flatten tensors to 1D for simplicity
    conv2d_kernel[(grid_size,)](
        input_ptr=input_flat.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr(),
        output_ptr=output_flat.data_ptr(),
        input_batch=B,
        input_channels=C_in,
        input_height=H_in,
        input_width=W_in,
        output_channels=C_out,
        kernel_height=K_H,
        kernel_width=K_W,
        output_height=H_out,
        output_width=W_out,
        stride_height=2,
        stride_width=2,
        pad_height=1,
        pad_width=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_M
    )
    
    return output_flat.reshape(B, C_out, H_out, W_out)

def replacement_func():
    return optimized_conv2d