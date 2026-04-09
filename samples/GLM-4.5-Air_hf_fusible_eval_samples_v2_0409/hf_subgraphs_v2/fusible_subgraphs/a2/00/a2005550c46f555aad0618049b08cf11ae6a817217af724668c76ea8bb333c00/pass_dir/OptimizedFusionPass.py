import torch
import triton
import triton.language as tl

# Pattern matching function to match the complete computation chain
def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    # BatchNorm: input=in_4, running_mean=in_0, running_var=in_1, weight=in_3, bias=in_2
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Element-wise addition with in_5
    tmp_5 = in_5 + tmp_4
    
    # ReLU activation 
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    
    # Mean reduction across spatial dimensions (2, 3) with keepdim=True
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    
    # Return the tuple that matches what the original function returns
    return (tmp_6, tmp_7)

def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)

@triton.jit
def optimized_fusion_kernel(
    # Input tensor pointers
    x_ptr, in_5_ptr,
    # BatchNorm parameters
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output tensor pointers
    relu_out_ptr, mean_sum_ptr,
    # Shape information
    batch_size, num_channels, height, width,
    # BatchNorm parameters
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * num_channels * height * width
    num_pid = tl.cdiv(total_elements, BLOCK_SIZE)
    
    if pid >= num_pid:
        return
    
    # Compute memory offsets for this thread block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert linear offset to 4D coordinates efficiently
    b = offsets // (num_channels * height * width)
    c = (offsets % (num_channels * height * width)) // (height * width)
    spatial = offsets % (height * width)
    h = spatial // width
    w = spatial % width
    
    # Load input tensors with optimized access patterns
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    in_5_val = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized BatchNorm parameter loading with conditional selection
    mean = tl.where(c < num_channels, tl.load(running_mean_ptr + c), 0.0)
    var = tl.where(c < num_channels, tl.load(running_var_ptr + c), 1.0)
    weight = tl.where(c < num_channels, tl.load(weight_ptr + c), 1.0)
    bias = tl.where(c < num_channels, tl.load(bias_ptr + c), 0.0)
    
    # Compute fused operations with minimal memory access
    var_plus_eps = var + eps
    inv_std = tl.math.rsqrt(var_plus_eps)
    x_norm = (x - mean) * inv_std
    x_bn = (x_norm * weight) + bias
    x_add = x_bn + in_5_val
    x_relu = tl.maximum(x_add, 0.0)
    
    # Store final ReLU output
    tl.store(relu_out_ptr + offsets, x_relu, mask=mask)
    
    # Concurrent mean calculation using atomic operations
    tl.atomic_add(mean_sum_ptr + c, x_relu)

@torch.fx.wrap
def optimized_fusion(in_4, in_0, in_1, in_3, in_2, in_5):
    batch_size, num_channels, height, width = in_4.shape
    
    # Allocate output tensors
    relu_out = torch.empty_like(in_5)
    mean_buffer = torch.zeros(num_channels, dtype=in_4.dtype, device=in_4.device)
    
    # Optimized block size for GPU efficiency
    BLOCK_SIZE = 1024
    
    # Launch optimized kernel
    total_elements = batch_size * num_channels * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_fusion_kernel[(num_programs,)](
        in_4, in_5,  # Input tensors
        in_0, in_1, in_3, in_2,  # BatchNorm parameters
        relu_out, mean_buffer,  # Output tensors
        batch_size, num_channels, height, width,
        1e-05,  # eps
        BLOCK_SIZE,
    )
    
    # Finalize mean computation
    scale = 1.0 / (height * width)
    mean_result = mean_buffer * scale
    mean_reshaped = mean_result.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_channels, 1, 1)
    
    return relu_out, mean_reshaped

def replacement_func():
    return optimized_fusion