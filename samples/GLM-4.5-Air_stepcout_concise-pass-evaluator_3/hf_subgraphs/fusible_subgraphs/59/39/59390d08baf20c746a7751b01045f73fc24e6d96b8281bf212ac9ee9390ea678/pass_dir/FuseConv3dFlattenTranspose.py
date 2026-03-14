import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for the complete computation sequence
    This matches the computation pattern from the model files that returns (tmp_5, tmp_7)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = torch.conv3d(in_3, tmp_1, tmp_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = tmp_3.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_2.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    return tmp_5, tmp_7

def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the replacement
    """
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv3d_flatten_transpose_kernel(
    input_ptr,      # Input tensor [N, C, D, H, W]
    weight_ptr,     # Weight tensor [O, C, kD, kH, kW] 
    bias_ptr,       # Bias [O]
    output_ptr,     # Output tensor [N, O_flat, D_out]
    N, C, D, H, W,  # Input dimensions
    O, kD, kH, kW,  # Output channels and kernel dimensions
    stride_D, stride_H, stride_W,
    pad_D, pad_H, pad_W,
    dilation_D, dilation_H, dilation_W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for conv3d + flatten + transpose operations
    Optimized for GPU performance with memory coalescing
    """
    pid = tl.program_id(0)
    total_elements = N * O * D  # Flat dimension after fusion
    
    # Calculate which output element this program handles
    linear_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < total_elements
    
    if not tl.any(mask):
        return
    
    # Convert linear index to output coordinates [N, O, D]
    D_out_idx = linear_idx % D
    O_idx = (linear_idx // D) % O
    N_idx = linear_idx // (O * D)
    
    # Calculate input coordinates for convolution
    # D_in = (D_out - 1) * stride_D + kD - 1 (using full convolution formula)
    D_in = D_out_idx * stride_D + (kD - 1) * dilation_D
    D_in_base = D_in - (kD - 1) * dilation_D
    
    # Determine the spatial range to load for convolution
    D_start = max(0, D_in_base)
    D_end = min(D, D_in_base + kD)
    
    # Initialize output accumulator
    output_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process valid kernel positions
    for kD_offset in range(0, kD):
        D_input = D_in_base + kD_offset * dilation_D
        if 0 <= D_input < D:
            # Load input slice [N, C, D_input]
            input_offset = N_idx * C * D + D_input
            input_slice = tl.load(input_ptr + input_offset * C + tl.arange(0, C), 
                                mask=tl.arange(0, C) < C, 
                                other=0.0).to(tl.float32)
            
            # Load weight slice [O, C, kD_offset]
            weight_offset = O_idx * C * kD + kD_offset
            weight_slice = tl.load(weight_ptr + weight_offset * C + tl.arange(0, C), 
                                 mask=tl.arange(0, C) < C, 
                                 other=0.0).to(tl.float32)
            
            # Convolution operation: input_slice (1,C) * weight_slice (1,C) -> scalar
            conv_val = tl.sum(input_slice * weight_slice, axis=0)
            
            # Add to accumulator
            output_val += conv_val
    
    # Load bias and add to result
    bias_val = tl.load(bias_ptr + O_idx, other=0.0)
    final_val = output_val + bias_val
    
    # Store result
    output_offset = N_idx * O * D + O_idx * D + D_out_idx
    tl.store(output_ptr + linear_idx, final_val, mask=mask)

@torch.fx.wrap
def fused_conv3d_flatten_transpose(bias, weight, position_embeddings, input_tensor):
    """
    Wrapper function for the fused conv3d + flatten + transpose + detach + type_as kernel
    Returns both conv3d output and type-converted position embeddings
    """
    # Get input dimensions
    N, C, D, H, W = input_tensor.shape
    O, _, kD, kH, kW = weight.shape
    
    # Calculate output spatial dimensions after convolution
    D_out = (D - kD) // 2 + 1  # stride_D = 2
    
    # Create output tensor for conv3d + flatten + transpose
    conv_output = torch.empty((N, O, D_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set block size for kernel
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    total_elements = N * O * D_out
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv3d_flatten_transpose_kernel[grid_size](
        input_tensor,
        weight, 
        bias,
        conv_output,
        N, C, D, H, W,
        O, kD, kH, kW,
        2, 16, 16,  # stride_D, stride_H, stride_W
        0, 0, 0,    # pad_D, pad_H, pad_W  
        1, 1, 1,    # dilation_D, dilation_H, dilation_W
        BLOCK_SIZE
    )
    
    # Handle detach + type_as operation
    detached_position = position_embeddings.detach()
    typed_position = detached_position.type_as(conv_output)
    
    return conv_output, typed_position

def replacement_func():
    """
    Return the replacement function (must be zero-argument)
    """
    return fused_conv3d_flatten_transpose