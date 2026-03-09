import torch
import triton
import triton.language as tl
import math

def pattern(conv_input, conv_weight, conv_bias, mult_input, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    # Direct computation without intermediate variables
    attention_weights = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    attention_weights = attention_weights.sigmoid()
    scaled_features = mult_input * attention_weights
    relu_output = torch.nn.functional.relu(scaled_features, inplace=True)
    bn_output = torch.nn.functional.batch_norm(relu_output, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    return relu_output, bn_output

def replacement_args(conv_in, conv_weight, conv_bias, mult_left, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    return (conv_in, conv_weight, conv_bias, mult_left, bn_running_mean, bn_running_var, bn_weight, bn_bias)

@triton.jit
def fused_se_bn_kernel(
    main_input_ptr, se_input_ptr, conv_weight_ptr, conv_bias_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    relu_out_ptr, bn_out_ptr,
    N, C, H, W, se_C,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one element
    pid = tl.program_id(0)
    
    # Calculate total elements and bounds
    total_elements = N * C * H * W
    if pid >= total_elements:
        return
    
    # Convert 1D index to 4D coordinates
    w = pid % W
    h = (pid // W) % H
    c = (pid // (W * H)) % C
    n = pid // (W * H * C)
    
    # Load main input element
    main_val = tl.load(main_input_ptr + n * C * H * W + c * H * W + h * W + w)
    
    # Load SE branch inputs (1x1 conv output + sigmoid)
    # For each spatial location, compute the attention weights
    se_offset = n * se_C  # se_input is [N, se_C, 1, 1]
    conv_offset = c * se_C  # conv_weight is [C, se_C, 1, 1]
    
    # Compute 1x1 convolution output
    conv_result = 0.0
    for se_c in range(se_C):
        se_val = tl.load(se_input_ptr + se_offset + se_c)
        weight_val = tl.load(conv_weight_ptr + conv_offset + se_c)
        conv_result += se_val * weight_val
    
    # Add bias
    conv_result += tl.load(conv_bias_ptr + c)
    
    # Sigmoid activation
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Apply attention scaling
    attended_val = main_val * sigmoid_val
    
    # ReLU activation
    relu_val = tl.maximum(attended_val, 0.0)
    
    # Store ReLU output
    tl.store(relu_out_ptr + pid, relu_val)
    
    # Load batch normalization parameters
    bn_mean = tl.load(bn_mean_ptr + c)
    bn_var = tl.load(bn_var_ptr + c)
    bn_weight = tl.load(bn_weight_ptr + c)
    bn_bias = tl.load(bn_bias_ptr + c)
    
    # Apply batch normalization
    bn_val = (relu_val - bn_mean) / tl.sqrt(bn_var + 1e-5)
    bn_val = bn_val * bn_weight + bn_bias
    
    # Store batch normalization output
    tl.store(bn_out_ptr + pid, bn_val)

@torch.fx.wrap
def fused_conv_se_bn(main_input, se_input, conv_weight, conv_bias, 
                    bn_running_mean, bn_running_var, bn_weight, bn_bias):
    # Get input shapes
    N, C, H, W = main_input.shape
    se_C = se_input.shape[1]
    
    # Create output tensors
    relu_out = torch.empty_like(main_input)
    bn_out = torch.empty_like(main_input)
    
    # Calculate grid size (one program per element)
    total_elements = N * C * H * W
    
    # Use reasonable block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_se_bn_kernel[num_programs](
        main_input, se_input, conv_weight, conv_bias,
        bn_running_mean, bn_running_var, bn_weight, bn_bias,
        relu_out, bn_out,
        N, C, H, W, se_C,
        BLOCK_SIZE
    )
    
    return relu_out, bn_out

def replacement_func():
    return fused_conv_se_bn