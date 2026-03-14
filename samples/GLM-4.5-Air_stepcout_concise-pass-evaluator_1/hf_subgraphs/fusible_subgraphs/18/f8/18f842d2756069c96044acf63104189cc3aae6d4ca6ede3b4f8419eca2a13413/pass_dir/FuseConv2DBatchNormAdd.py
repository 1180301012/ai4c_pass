import torch
import triton
import triton.language as tl

def pattern(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    """
    Pattern matching for Conv2D + BatchNorm + Add operations
    in_6: input tensor
    in_4: conv weight  
    in_0: bn running_mean
    in_1: bn running_var
    in_3: bn weight
    in_2: bn bias
    in_5: residual tensor to add
    """
    tmp_5 = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6

def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5)

@triton.jit
def conv2d_batch_norm_add_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr, 
    bn_weight_ptr, bn_bias_ptr, residual_ptr, output_ptr,
    batch_size, input_channels, output_channels, 
    input_height, input_width, conv_weight_height, conv_weight_width,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Each program computes one tile of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the range of the tile each program will compute
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    # Create offset pointers for input, weight, and output
    off_input_m = start_m * stride_h + (conv_weight_height - 1) * dilation_h // 2 - padding_h
    off_input_n = start_n * stride_w + (conv_weight_width - 1) * dilation_w // 2 - padding_w
    
    off_input = (off_input_m * input_width + off_input_n) * input_channels
    off_weight = start_n * output_channels * conv_weight_height * conv_weight_width
    off_output = (start_m * input_width + start_n) * output_channels
    
    # Preload fused parameters
    eps_f = tl.load(eps)
    batch_norm_weight = tl.load(bn_weight_ptr + pid_n)
    running_mean_val = tl.load(running_mean_ptr + pid_n)
    running_var_val = tl.load(running_var_ptr + pid_n)
    bn_bias_val = tl.load(bn_bias_ptr + pid_n)
    
    # Precompute batch normalization scaling factors
    inv_std = 1.0 / tl.sqrt(running_var_val + eps_f)
    scale = batch_norm_weight * inv_std
    bias = bn_bias_val - running_mean_val * scale
    
    # Loop over batch
    for b in range(0, batch_size, 1):
        # Compute partial output
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Loop over input channels (channels are fused in groups)
        for k in range(0, input_channels, BLOCK_SIZE_K):
            # Load input block
            input_offsets = off_input + (b * input_height * input_width + k) * input_height * input_width
            input_block = tl.load(input_ptr + input_offsets, mask=None, other=0.0)
            
            # Load weight block
            weight_offsets = off_weight + k * conv_weight_height * conv_weight_width
            weight_block = tl.load(weight_ptr + weight_offsets, mask=None, other=0.0)
            
            # Convolution computation
            for i in range(BLOCK_SIZE_M):
                for j in range(BLOCK_SIZE_N):
                    for ki in range(conv_weight_height):
                        for kj in range(conv_weight_width):
                            input_val = tl.load(input_ptr + input_offsets + 
                                             i * input_height * input_channels + 
                                             j * input_channels + 
                                             (ki * dilation_h) * input_height * input_channels + 
                                             (kj * dilation_w) * input_channels, 
                                             mask=None, other=0.0)
                            weight_val = tl.load(weight_ptr + weight_offsets + 
                                               output_channels * (ki * conv_weight_width + kj), 
                                               mask=None, other=0.0)
                            acc[i, j] += input_val * weight_val
        
        # Apply batch normalization and add residual
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_N):
                residual_val = 0.0
                if start_m + i < batch_size and start_n + j < input_width:
                    residual_offsets = ((b * input_height + start_m + i) * input_width + start_n + j) * output_channels
                    residual_val = tl.load(residual_ptr + residual_offsets + pid_n * output_channels, mask=None, other=0.0)
                
                output_val = acc[i, j] * scale + bias + residual_val
                tl.store(output_ptr + off_output + (b * input_height * input_width + i * input_width + j) * output_channels + pid_n, output_val)

@torch.fx.wrap  
def fused_conv2d_batch_norm_add(input, weight, running_mean, running_var, bn_weight, bn_bias, residual):
    """Fused Conv2D + BatchNorm + Add using Triton kernel"""
    
    # Get tensor shapes
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, _, conv_weight_height, conv_weight_width = weight.shape
    
    # Create output tensor
    output = torch.zeros((batch_size, output_channels, input_height, input_width), 
                        dtype=input.dtype, device=input.device)
    
    # Define kernel parameters
    eps = 1e-05  # This matches the model's epsilon value
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (input_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    conv2d_batch_norm_add_kernel[(
        grid_m, 
        output_channels
    )](
        input_ptr=input,
        weight_ptr=weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        residual_ptr=residual,
        output_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        input_height=input_height,
        input_width=input_width,
        conv_weight_height=conv_weight_height,
        conv_weight_width=conv_weight_width,
        stride_h=1,
        stride_w=1,
        padding_h=0,
        padding_w=0,
        dilation_h=1,
        dilation_w=1,
        groups=1,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return output

def replacement_func():
    return fused_conv2d_batch_norm_add