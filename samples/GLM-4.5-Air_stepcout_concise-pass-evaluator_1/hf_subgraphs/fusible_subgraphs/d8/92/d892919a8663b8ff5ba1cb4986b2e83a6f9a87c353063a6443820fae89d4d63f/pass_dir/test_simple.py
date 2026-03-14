import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    # Batch normalization pattern exactly as in original computation
    batch_norm_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=0.001)
    
    # Return the batch norm output
    return batch_norm_out

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.jit
def optimized_batch_norm_kernel(
    input_ptr, output_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-3
):
    # Total number of elements in the tensor
    total_elements = N * C * H * W
    
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    
    # Calculate element offset
    element_offset = block_start + tl.arange(0, block_size)
    mask = element_offset < total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + element_offset, mask=mask, other=0.0)
    
    # Load batch norm parameters for each channel
    c_indices = element_offset % C
    
    # Load parameters with broadcasting to all elements for each channel
    running_mean_data = tl.load(running_mean_ptr + c_indices, mask=mask, other=0.0)
    running_var_data = tl.load(running_var_ptr + c_indices, mask=mask, other=1.0)
    weight_data = tl.load(weight_ptr + c_indices, mask=mask, other=1.0)
    bias_data = tl.load(bias_ptr + c_indices, mask=mask, other=0.0)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (input_data - running_mean_data) / tl.sqrt(running_var_data + EPS) * weight_data + bias_data
    
    # Store the result
    tl.store(output_ptr + element_offset, normalized, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    N, C, H, W = input_tensor.shape
    total_elements = N * C * H * W
    
    # Choose optimal block size based on tensor size
    if total_elements > 1024 * 1024:
        BLOCK_SIZE = 4096
    elif total_elements > 1024:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch the optimized kernel
    optimized_batch_norm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    def wrapper(input_tensor, running_mean, running_var, weight, bias):
        return optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias)
    
    return wrapper