import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    # Simplified pattern - just batch norm
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_4

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def optimized_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    num_features,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output
    pid = tl.program_id(0)
    
    # Calculate which element this program handles
    batch_stride = num_features * height * width
    
    # Calculate indices for this element
    batch_idx = pid // batch_stride
    feature_idx = (pid // (height * width)) % num_features
    spatial_idx = pid % (height * width)
    
    # Calculate input offset
    input_offset = batch_idx * batch_stride + feature_idx * (height * width) + spatial_idx
    
    # Load input element and parameters
    x = tl.load(input_ptr + input_offset, mask=True)
    mean = tl.load(running_mean_ptr + feature_idx)
    var = tl.load(running_var_ptr + feature_idx)
    weight_val = tl.load(weight_ptr + feature_idx)
    bias_val = tl.load(bias_ptr + feature_idx)
    
    # Cast to fp32 for precision in math operations (required for rsqrt)
    x_f32 = tl.cast(x, tl.float32)
    mean_f32 = tl.cast(mean, tl.float32)
    var_f32 = tl.cast(var, tl.float32)
    weight_f32 = tl.cast(weight_val, tl.float32)
    bias_f32 = tl.cast(bias_val, tl.float32)
    eps_f32 = tl.cast(eps, tl.float32)
    
    # Apply batch normalization in fp32 for maximum precision
    x_sub_mean = x_f32 - mean_f32
    var_plus_eps = var_f32 + eps_f32
    
    # Compute reciprocal square root in fp32
    inv_std = tl.math.rsqrt(var_plus_eps)
    
    # Apply scaling and shifting (PyTorch BN math)
    normalized_f32 = x_sub_mean * inv_std * weight_f32 + bias_f32
    
    # Cast back to original dtype with proper rounding
    if x.type.scalar.name == 'bfloat16':
        # For bfloat16, use appropriate rounding
        normalized_f16 = tl.cast(normalized_f32, tl.float16)
        output_val = tl.cast(normalized_f16, tl.bfloat16)
    else:
        # For float16 and float32, direct cast is fine
        output_val = tl.cast(normalized_f32, x.type)
    
    # Store result
    tl.store(output_ptr + input_offset, output_val, mask=True)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    """
    Optimized batch normalization operation with improved memory access
    """
    batch_size, num_features, height, width = input_tensor.shape
    
    # Calculate optimal block size and grid
    BLOCK_SIZE = 1024  # Optimal for most GPU architectures
    total_elements = batch_size * num_features * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with the same dtype as input
    output = torch.empty_like(input_tensor)
    
    # Launch optimized kernel
    optimized_batch_norm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        num_features=num_features,
        height=height,
        width=width,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_batch_norm