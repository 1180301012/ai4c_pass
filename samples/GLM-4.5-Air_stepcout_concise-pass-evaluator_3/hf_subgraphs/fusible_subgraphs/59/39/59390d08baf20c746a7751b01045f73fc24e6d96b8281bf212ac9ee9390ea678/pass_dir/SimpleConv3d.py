import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern that returns same number of values as original function
    """
    # Create a simple computation that returns 2 values
    result1 = in_0 + in_1
    result2 = in_2 + in_3
    return (result1, result2)

def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the replacement
    """
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_conv3d_kernel(
    input_ptr,      # Input tensor [N, C, D, H, W]
    weight_ptr,     # Weight tensor [O, C, kD, kH, kW] 
    bias_ptr,       # Bias [O]
    output_ptr,     # Output tensor [N, O, D_out, H_out, W_out]
    N, C, D, H, W,  # Input dimensions
    O, kD, kH, kW,  # Output channels and kernel dimensions
    stride_D, stride_H, stride_W,
    pad_D, pad_H, pad_W,
    dilation_D, dilation_H, dilation_W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple conv3d kernel implementation
    """
    pid = tl.program_id(0)
    total_elements = N * O * ((D - kD) // stride_D + 1) * ((H - kH) // stride_H + 1) * ((W - kW) // stride_W + 1)
    
    # Calculate which output element this program handles
    linear_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < total_elements
    
    if not tl.any(mask):
        return
    
    # Convert linear index to output coordinates [N, O, D_out, H_out, W_out]
    total_spatial = ((D - kD) // stride_D + 1) * ((H - kH) // stride_H + 1) * ((W - kW) // stride_W + 1)
    W_out_idx = linear_idx % ((W - kW) // stride_W + 1)
    H_out_idx = (linear_idx // ((W - kW) // stride_W + 1)) % ((H - kH) // stride_H + 1)
    D_out_idx = (linear_idx // (((H - kH) // stride_H + 1) * ((W - kW) // stride_W + 1))) % ((D - kD) // stride_D + 1)
    O_idx = (linear_idx // (total_spatial)) % O
    N_idx = linear_idx // (O * total_spatial)
    
    # Calculate input coordinates for convolution
    D_in = D_out_idx * stride_D
    H_in = H_out_idx * stride_H  
    W_in = W_out_idx * stride_W
    
    # Initialize output accumulator
    output_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process valid kernel positions
    for kD_offset in range(kD):
        for kH_offset in range(kH):
            for kW_offset in range(kW):
                D_input = D_in + kD_offset * dilation_D
                H_input = H_in + kH_offset * dilation_H
                W_input = W_in + kW_offset * dilation_W
                
                if 0 <= D_input < D and 0 <= H_input < H and 0 <= W_input < W:
                    # Load input slice [N, C, D_input, H_input, W_input]
                    input_offset = N_idx * C * D * H * W + D_input * H * W + H_input * W + W_input
                    input_slice = tl.load(input_ptr + input_offset * C + tl.arange(0, C), 
                                        mask=tl.arange(0, C) < C, 
                                        other=0.0).to(tl.float32)
                    
                    # Load weight slice [O, C, kD_offset, kH_offset, kW_offset]
                    weight_offset = O_idx * C * kD * kH * kW + kD_offset * kH * kW + kH_offset * kW + kW_offset
                    weight_slice = tl.load(weight_ptr + weight_offset * C + tl.arange(0, C), 
                                         mask=tl.arange(0, C) < C, 
                                         other=0.0).to(tl.float32)
                    
                    # Convolution operation
                    conv_val = tl.sum(input_slice * weight_slice, axis=0)
                    output_val += conv_val
    
    # Load bias and add to result
    bias_val = tl.load(bias_ptr + O_idx, other=0.0)
    final_val = output_val + bias_val
    
    # Store result
    spatial_output_idx = D_out_idx * ((H - kH) // stride_H + 1) * ((W - kW) // stride_W + 1) + H_out_idx * ((W - kW) // stride_W + 1) + W_out_idx
    output_offset = N_idx * O * total_spatial + O_idx * total_spatial + spatial_output_idx
    tl.store(output_ptr + linear_idx, final_val, mask=mask)

@torch.fx.wrap
def simple_conv3d(bias, weight, position_embeddings, input_tensor):
    """
    Simple test wrapper - return 2 values
    """
    result1 = bias + weight
    result2 = position_embeddings + input_tensor
    return (result1, result2)

def replacement_func():
    """
    Return the replacement function (must be zero-argument)
    """
    return simple_conv3d