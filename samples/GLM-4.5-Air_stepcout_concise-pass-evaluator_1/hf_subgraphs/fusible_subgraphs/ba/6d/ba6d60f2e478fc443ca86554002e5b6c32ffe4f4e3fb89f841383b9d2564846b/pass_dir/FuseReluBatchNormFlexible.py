import torch
import triton
import triton.language as tl

def pattern(relu_input, _unused1, _unused2, _unused3, _unused4, _unused5, _unused6, _unused7):
    # Flexible ReLU + BatchNorm sequence
    relu_result = torch.nn.functional.relu(relu_input, inplace=True)
    # Try to match batch norm with flexible argument order
    # The actual arguments will vary, so we need to extract them from the match context
    batch_norm_result = torch.nn.functional.batch_norm(relu_result, _unused1, _unused2, _unused3, _unused4, False, 0.1, 1e-05)
    # Return both outputs as in the original model
    return relu_result, batch_norm_result

def replacement_args(relu_input, _unused1, _unused2, _unused3, _unused4, _unused5, _unused6, _unused7):
    return (relu_input, _unused1, _unused2, _unused3, _unused4, _unused5, _unused6, _unused7)

@triton.jit
def relu_fused_bn_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    relu_output_ptr,
    n_batch, n_channels, height, width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program handles a 2D tile of the batch-channel space
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute memory addresses
    input_base = pid_b * n_channels * height * width + pid_c * height * width
    output_base = pid_b * n_channels * height * width + pid_c * height * width
    relu_output_base = pid_b * n_channels * height * width + pid_c * height * width
    
    # Load input data
    input_ptr_local = input_ptr + input_base
    input_data = tl.load(input_ptr_local, mask=input_ptr_local < input_base + height * width, other=0.0)
    
    # Load BatchNorm parameters
    bn_mean = tl.load(running_mean_ptr + pid_c)
    bn_var = tl.load(running_var_ptr + pid_c)
    bn_weight_val = tl.load(weight_ptr + pid_c)
    bn_bias_val = tl.load(bias_ptr + pid_c)
    
    # Apply BatchNorm normalization
    norm_data = (input_data - bn_mean) / tl.sqrt(bn_var + eps)
    bn_data = norm_data * bn_weight_val + bn_bias_val
    
    # Apply ReLU activation
    relu_data = tl.max(bn_data, 0.0)
    
    # Store results
    output_ptr_local = output_ptr + output_base
    relu_output_ptr_local = relu_output_ptr + relu_output_base
    
    tl.store(output_ptr_local, bn_data, mask=output_ptr_local < output_base + height * width)
    tl.store(relu_output_ptr_local, relu_data, mask=relu_output_ptr_local < relu_output_base + height * width)

@torch.fx.wrap
def fused_relu_batch_norm(input_tensor, running_mean, running_var, bn_weight, bn_bias):
    # Get input shapes
    n_batch, n_channels, height, width = input_tensor.shape
    
    # Create output tensors
    bn_output = torch.empty_like(input_tensor, dtype=torch.float32)
    relu_output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Block size for efficient GPU utilization
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid_b = n_batch
    grid_c = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    relu_fused_bn_kernel[grid_b, grid_c](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=bn_weight,
        bias_ptr=bn_bias,
        output_ptr=bn_output,
        relu_output_ptr=relu_output,
        n_batch=n_batch,
        n_channels=n_channels,
        height=height,
        width=width,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return relu_output, bn_output

def replacement_func():
    return fused_relu_batch_norm