import torch
import triton
import triton.language as tl

def pattern(in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2):
    # Full SE attention + ReLU + BatchNorm sequence
    tmp_6 = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1), 1)
    tmp_7 = tmp_6.sigmoid()
    tmp_8 = in_6 * tmp_7
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
    tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # Return both outputs as in the original model
    return tmp_9, tmp_10

def replacement_args(in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2):
    return (in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2)

@triton.jit
def full_se_attention_kernel(
    se_input_ptr,
    se_weight_ptr,
    se_bias_ptr,
    feature_map_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    relu_output_ptr,
    bn_output_ptr,
    n_batch, n_in_channels, n_out_channels, feature_height, feature_width,
    conv_height, conv_width,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Program handles batch and output channel space
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # SE processing (small spatial dimensions)
    # Load SE input: [N, C_in, H, W] -> typically [N, 20, 1, 1]
    se_input_base = pid_b * n_in_channels * conv_height * conv_width
    if conv_height == 1 and conv_width == 1:
        se_input_val = tl.load(se_input_ptr + se_input_base + pid_c)
    else:
        se_input_val = tl.load(se_input_ptr + se_input_base + pid_c * conv_height * conv_width)
    
    # Load SE weight and bias: [C_out, C_in, 1, 1]
    se_weight_base = pid_c * n_in_channels
    se_weight_val = tl.load(se_weight_ptr + se_weight_base + pid_c)
    se_bias_val = tl.load(se_bias_ptr + pid_c)
    
    # SE attention: conv2d -> sigmoid -> multiply
    se_conv_result = se_input_val * se_weight_val + se_bias_val
    se_sigmoid_result = 1.0 / (1.0 + tl.exp(-se_conv_result))
    
    # Load and multiply with feature map: [N, C_out, H, W]
    feature_map_base = pid_b * n_out_channels * feature_height * feature_width
    feature_map_val_local = feature_map_ptr + feature_map_base + pid_c * feature_height * feature_width
    
    # Store ReLU output (first output)
    relu_base = pid_b * n_out_channels * feature_height * feature_width + pid_c * feature_height * feature_width
    relu_ptr_local = relu_output_ptr + relu_base
    
    # Store BN output (second output)
    bn_base = pid_b * n_out_channels * feature_height * feature_width + pid_c * feature_height * feature_width  
    bn_ptr_local = bn_output_ptr + bn_base
    
    # Load BatchNorm parameters
    bn_mean = tl.load(running_mean_ptr + pid_c)
    bn_var = tl.load(running_var_ptr + pid_c)
    bn_weight_val = tl.load(bn_weight_ptr + pid_c)
    bn_bias_val = tl.load(bn_bias_ptr + pid_c)
    
    # Process all spatial positions for the current batch and channel
    for h in range(feature_height):
        for w in range(feature_width):
            feature_addr = feature_map_val_local + h * feature_width + w
            relu_addr = relu_ptr_local + h * feature_width + w
            bn_addr = bn_ptr_local + h * feature_width + w
            
            # Load feature map value (broadcasted attention across spatial dimensions)
            feature_map_val = tl.load(feature_map_val_local + h * feature_width + w)
            attention_val = se_sigmoid_result
            
            # Apply SE attention
            se_activated = feature_map_val * attention_val
            
            # Apply BatchNorm
            bn_norm_data = (se_activated - bn_mean) / tl.sqrt(bn_var + eps)
            bn_out = bn_norm_data * bn_weight_val + bn_bias_val
            
            # Apply ReLU
            relu_out = tl.max(bn_out, 0.0)
            
            # Store results
            tl.store(bn_addr, bn_out)
            tl.store(relu_addr, relu_out)

@torch.fx.wrap  
def full_se_attention_fusion(se_input, se_weight, se_bias, feature_map, running_mean, running_var, bn_weight, bn_bias):
    # Get input shapes
    n_batch, n_in_channels, se_height, se_width = se_input.shape
    n_out_channels = se_weight.shape[0]  # [C_out, C_in, 1, 1]
    feature_channels, feature_height, feature_width = feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]
    
    # Create output tensors
    relu_output = torch.empty_like(feature_map, dtype=torch.float32)
    bn_output = torch.empty_like(feature_map, dtype=torch.float32)
    
    # Optimize block sizes for GPU
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_C = 64
    
    # Calculate grid dimensions
    grid_n = (n_batch + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (n_out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    full_se_attention_kernel[grid_n, grid_c](
        se_input_ptr=se_input,
        se_weight_ptr=se_weight,
        se_bias_ptr=se_bias,
        feature_map_ptr=feature_map,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        relu_output_ptr=relu_output,
        bn_output_ptr=bn_output,
        n_batch=n_batch,
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        feature_height=feature_height,
        feature_width=feature_width,
        conv_height=se_height,
        conv_width=se_width,
        eps=1e-05,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return relu_output, bn_output

def replacement_func():
    return full_se_attention_fusion