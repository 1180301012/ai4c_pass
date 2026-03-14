import torch
import triton
import triton.language as tl
import math

def pattern(in_5, tmp_6, tmp_1, tmp_2, tmp_4, tmp_3):
    # Concat operation - exactly as in the graphs
    tmp_7 = torch.cat([in_5, tmp_6], 1)
    # Batch norm operation - exactly as in the graphs  
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    # ReLU operation - exactly as in the graphs
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return tmp_7, tmp_8, tmp_9

def replacement_args(in_5, tmp_6, tmp_1, tmp_2, tmp_4, tmp_3):
    return (in_5, tmp_6, tmp_1, tmp_2, tmp_4, tmp_3)

@triton.jit
def fused_cat_bn_relu_kernel(
    input_tensor_ptr,
    target_shape_tensor_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    input_n,
    concat_c,
    input_h,
    input_w,
    target_c,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Program id
    pid = tl.program_id(0)
    
    # Output coordinates
    out_c_idx = pid // (input_h * input_w)
    out_h_idx = (pid % (input_h * input_w)) // input_w
    out_w_idx = pid % input_w
    
    # Get source channel index
    if out_c_idx >= concat_c:
        # From target_shape_tensor (second part of concatenation)
        source_c_idx = out_c_idx - concat_c
        tensor_idx = 1
        src_ptr = target_shape_tensor_ptr
        src_c = target_c
    else:
        # From input_tensor (first part of concatenation)
        source_c_idx = out_c_idx
        tensor_idx = 0
        src_ptr = input_tensor_ptr
        src_c = concat_c - target_c
    
    # Calculate input index
    batch_idx = tl.program_id(1)
    channel_idx = out_c_idx
    
    # Check bounds
    if batch_idx >= input_n or channel_idx >= concat_c or out_h_idx >= input_h or out_w_idx >= input_w:
        return
    
    # Load input value
    input_idx = batch_idx * concat_c * input_h * input_w + channel_idx * input_h * input_w + out_h_idx * input_w + out_w_idx
    input_val = tl.load(src_ptr + input_idx)
    
    # Load batch norm parameters
    bn_params_idx = channel_idx
    running_mean_val = tl.load(running_mean_ptr + bn_params_idx)
    running_var_val = tl.load(running_var_ptr + bn_params_idx)
    weight_val = tl.load(weight_ptr + bn_params_idx)
    bias_val = tl.load(bias_ptr + bn_params_idx)
    
    # Batch norm formula: (x - running_mean) / sqrt(running_var + eps) * weight + bias
    denom = tl.sqrt(running_var_val + eps)
    normalized = (input_val - running_mean_val) / denom
    batch_norm_result = normalized * weight_val + bias_val
    
    # ReLU operation
    relu_result = tl.math.max(batch_norm_result, 0.0)
    
    # Store result
    output_idx = batch_idx * concat_c * input_h * input_w + out_c_idx * input_h * input_w + out_h_idx * input_w + out_w_idx
    tl.store(output_ptr + output_idx, relu_result)

@torch.fx.wrap
def fused_cat_bn_relu(in_5, tmp_6, tmp_1, tmp_2, tmp_4, tmp_3):
    # Get input shapes
    input_n, input_c1, input_h, input_w = in_5.shape
    target_c = tmp_6.shape[1]
    concat_c = input_c1 + target_c
    
    # Create output tensor
    output = torch.empty((input_n, concat_c, input_h, input_w), dtype=in_5.dtype, device=in_5.device)
    
    # Block sizes for kernel
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    
    # Calculate grid size
    total_elements = input_n * concat_c * input_h * input_w
    grid_size = (total_elements + 256 - 1) // 256
    
    # Launch the fused kernel
    fused_cat_bn_relu_kernel[(
        (total_elements + 255) // 256,
        input_n,
        concat_c
    )](
        in_5,
        tmp_6,
        output,
        tmp_1,
        tmp_2,
        tmp_4,
        tmp_3,
        input_n,
        concat_c,
        input_h,
        input_w,
        target_c,
        0.1,  # hardcoded momentum from the graphs
        0.001,  # hardcoded eps from the graphs
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
    )
    
    # For the concat operation, the optimized kernel handles the concatenation internally
    # Return the concatenated intermediate using a separate tensor
    concat_result = torch.empty((input_n, concat_c, input_h, input_w), dtype=in_5.dtype, device=in_5.device)
    # Copy the concatenation result (first input_tensor part remains unchanged, second part from target_shape_tensor)
    concat_result[:, :input_c1] = in_5
    concat_result[:, input_c1:] = tmp_6
    return concat_result, output, output

def replacement_func():
    return fused_cat_bn_relu