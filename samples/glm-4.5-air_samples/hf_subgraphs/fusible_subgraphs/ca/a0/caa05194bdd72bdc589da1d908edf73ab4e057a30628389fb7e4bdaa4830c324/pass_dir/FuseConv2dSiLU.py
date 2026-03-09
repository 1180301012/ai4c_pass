import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weights):
    # Simple: conv2d followed by silu - try with 2 arguments
    conv_out = torch.conv2d(conv_input, conv_weights, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    silu_out = torch.nn.functional.silu(conv_out, inplace=False)
    return conv_out, silu_out

def replacement_args(conv_input, conv_weights):
    return (conv_input, conv_weights)

# Optimized fused Conv2D + SiLU kernel
@triton.jit
def conv2d_silu_kernel(
    input_ptr, input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_ptr, weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    bias_ptr, bias_stride_0,
    output_ptr, output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    N, C_in, H_in, W_in, C_out, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Output feature map position
    m = pid % ((H_in * W_in + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    n_block = pid // ((H_in * W_in + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    # Compute output position
    h_out = m * BLOCK_SIZE_M
    w_out = (n_block % W_in)
    c_out = n_block // W_in
    
    # Ensure we're within bounds
    if h_out >= H_in or w_out >= W_in or c_out >= C_out:
        return
    
    # Load bias
    bias_val = tl.load(bias_ptr + c_out * bias_stride_0)
    
    # Initialize accumulator
    acc = bias_val
    
    # Loop over input channels
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, C_in)
        
        # Compute input position for this block
        h_base = h_out
        w_base = w_out
        
        # Load weight block
        weight_offset = c_out * weight_stride_0 + k * weight_stride_1
        weight_block = tl.load(weight_ptr + weight_offset)
        
        # Load input block
        input_offset = k * input_stride_0 + h_base * input_stride_2 + w_base * input_stride_3
        input_val = tl.load(input_ptr + input_offset)
        
        # Accumulate convolution
        acc += weight_block * input_val
    
    # Apply SiLU activation: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    x = acc
    exp_neg_x = tl.exp(-tl.abs(x))
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    if x < 0:
        sigmoid_x *= exp_neg_x
    
    output_val = x * sigmoid_x
    
    # Store output
    output_offset = c_out * output_stride_0 + h_base * output_stride_2 + w_base * output_stride_3
    tl.store(output_ptr + output_offset, output_val)



@torch.fx.wrap
def conv2d_silu_fused(conv_input, conv_weights):
    # Get input and output shapes
    N, C_in, H_in, W_in = conv_input.shape
    C_out = conv_weights.shape[0]
    
    # Create output tensor
    output = torch.empty((N, C_out, H_in, W_in), dtype=conv_input.dtype, device=conv_input.device)
    
    # Simple block sizes for this specific case
    BLOCK_SIZE_M = 4  # Height/block
    BLOCK_SIZE_N = 64  # Width/output_channels per block
    
    # Calculate grid size: total output positions
    total_positions = H_in * W_in * C_out
    grid_size = (total_positions + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    if grid_size > 0:
        conv2d_silu_kernel[grid_size](
            input_ptr=conv_input,
            input_stride_0=conv_input.stride(0), input_stride_1=conv_input.stride(1), 
            input_stride_2=conv_input.stride(2), input_stride_3=conv_input.stride(3),
            weight_ptr=conv_weights,
            weight_stride_0=conv_weights.stride(0), weight_stride_1=conv_weights.stride(1),
            weight_stride_2=conv_weights.stride(2), weight_stride_3=conv_weights.stride(3),
            bias_ptr=torch.zeros(C_out, dtype=conv_weights.dtype, device=conv_weights.device),  # Assume no bias for now
            bias_stride_0=1,
            output_ptr=output,
            output_stride_0=output.stride(0), output_stride_1=output.stride(1),
            output_stride_2=output.stride(2), output_stride_3=output.stride(3),
            N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=1
        )
    
    return output

# Return the fused function
def replacement_func():
    return conv2d_silu_fused