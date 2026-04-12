import torch
import triton
import triton.language as tl

# Simple pattern matching just for conv2d operation
def pattern(input_tensor, weight_tensor, bias_tensor):
    # Simple conv2d pattern to test if basic matching works
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    return conv_result

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Optimized Triton kernel for fused conv2d + sigmoid + multiply
@triton.jit
def fused_conv_sigmoid_mul_kernel(
    input_ptr, weight_ptr, bias_ptr, multiply_ptr, output_ptr,
    N, O, I, H_in, W_in, H_out, W_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr
):
    # Calculate program indices
    pid = tl.program_id(0)
    n_offset = pid % N
    o_offset = (pid // N) % O
    
    # Calculate output spatial coordinates
    h_offset = ((pid // (N * O)) % H_out)
    w_offset = (pid // (N * O * H_out))
    
    if h_offset >= H_out or w_offset >= W_out:
        return
    
    # Calculate input coordinates with padding
    in_h = h_offset * STRIDE_H - PAD_H
    in_w = w_offset * STRIDE_W - PAD_W
    
    # Initialize output
    output = tl.zeros((1,), dtype=tl.float16)
    
    # Loop over input channels
    for i in range(0, I, BLOCK_SIZE_I):
        i_block = min(i + BLOCK_SIZE_I, I)
        
        # Load input block (with bounds checking)
        if in_h >= 0 and in_h < H_in and in_w >= 0 and in_w < W_in:
            # Load input slice for current (n, h, w) and input channel block
            input_base = input_ptr + n_offset * I * H_in * W_in + i * H_in * W_in + in_h * W_in + in_w
            input_vals = tl.load(input_base + tl.arange(0, i_block - i), mask=(i_block - i > 0), other=0.0)
            
            # Load weight slice for current output channel and input channel block
            weight_base = weight_ptr + o_offset * I * 1 * 1 + i * 1 * 1
            weight_vals = tl.load(weight_base + tl.arange(0, i_block - i), mask=(i_block - i > 0), other=0.0)
            
            # Convolution operation
            conv_val = tl.sum(input_vals * weight_vals)
            
            # Add bias
            bias_base = bias_ptr + o_offset
            bias_val = tl.load(bias_base)
            conv_val = conv_val + bias_val
        else:
            conv_val = bias_ptr + o_offset
            conv_val = tl.load(conv_val)
        
        # Accumulate convolution result
        output = output + conv_val
    
    # Apply sigmoid and multiply
    conv_result = output
    sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Load multiplication tensor value (broadcasted from [N, O_out, H_out, W_out])
    multiply_base = multiply_ptr + n_offset * O * H_out * W_out + o_offset * H_out * W_out + h_offset * W_out + w_offset
    multiply_val = tl.load(multiply_base)
    
    # Final multiply
    final_result = multiply_val * sigmoid_result
    
    # Store sigmoid result and final result
    sigmoid_ptr = output_ptr + n_offset * O * H_out * W_out + o_offset * H_out * W_out + h_offset * W_out + w_offset
    mul_output_ptr = sigmoid_ptr + N * O * H_out * W_out
    
    tl.store(sigmoid_ptr, sigmoid_result)
    tl.store(mul_output_ptr, final_result)

# Wrapper function for the fused operation
@torch.fx.wrap
def fused_conv_sigmoid_mul(input_tensor, weight_tensor, bias_tensor, multiply_tensor):
    # Get tensor shapes
    N, C_in, H_in, W_in = input_tensor.shape
    O, I, KH, KW = weight_tensor.shape
    C_out = bias_tensor.shape[0]
    
    # For 1x1 convolution with stride 1, padding 0, dilation 1
    H_out = H_in
    W_out = W_in
    
    # Create combined output buffer: [N, C_out, H_out, W_out] for sigmoid + [N, C_out, H_out, W_out] for multiply
    combined_output = torch.empty((2, N, C_out, H_out, W_out), dtype=torch.float16, device=input_tensor.device)
    sigmoid_output = combined_output[0]
    mul_output = combined_output[1]
    
    # Calculate grid size
    total_elements = N * C_out * H_out * W_out
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with contiguous memory layout
    fused_output = torch.empty((2, N, C_out, H_out, W_out), dtype=torch.float16, device=input_tensor.device)
    fused_conv_sigmoid_mul_kernel[grid_size](
        input_tensor, weight_tensor, bias_tensor, multiply_tensor,
        fused_output, N, C_out, C_in, H_in, W_in, H_out, W_out,
        32, 32, 32, 1, 1, 0, 0, 1, 1
    )
    
    sigmoid_output = fused_output[0]
    mul_output = fused_output[1]
    
    return sigmoid_output, mul_output

# Simple Triton kernel for conv2d
@triton.jit
def simple_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_out, C_in, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_offset = pid % N
    c_offset = (pid // N) % C_out
    
    if c_offset >= C_out:
        return
        
    # Initialize output as scalar
    output = 0.0
    
    # Simple 1x1 convolution for each spatial position
    for h in range(H):
        for w in range(W):
            # Compute convolution at position (h, w)
            conv_val = 0.0
            for ci in range(C_in):
                # Load input value at (n, ci, h, w)
                input_val = tl.load(input_ptr + n_offset * C_in * H * W + ci * H * W + h * W + w)
                # Load weight value at (c_offset, ci, 0, 0) for 1x1 conv
                weight_val = tl.load(weight_ptr + c_offset * C_in * 1 * 1 + ci * 1 * 1)
                conv_val = conv_val + input_val * weight_val
            
            # Add bias
            bias_val = tl.load(bias_ptr + c_offset)
            final_val = conv_val + bias_val
            
            # Store result at (n, c_offset, h, w)
            tl.store(output_ptr + n_offset * C_out * H * W + c_offset * H * W + h * W + w, final_val)

@torch.fx.wrap
def simple_conv2d_optimized(input_tensor, weight_tensor, bias_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out, _, _, _ = weight_tensor.shape
    
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    total_elements = N * C_out
    grid_size = ((total_elements + 1023) // 1024,)
    
    simple_conv2d_kernel[grid_size](
        input_tensor, weight_tensor, bias_tensor, output,
        N, C_out, C_in, H, W, 1024
    )
    
    return output

# Replacement function (returns the optimized function reference)
def replacement_func():
    return simple_conv2d_optimized