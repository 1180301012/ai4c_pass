import torch
import triton
import triton.language as tl

def pattern(concatenated_input, running_mean, running_var, weight, bias, prelu_weight):
    # Batch normalization exactly as in original
    batch_norm_out = torch.nn.functional.batch_norm(concatenated_input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=0.001)
    
    # PReLU activation 
    prelu_out = torch.prelu(batch_norm_out, prelu_weight)
    
    # Return exactly what the original graph returns
    return batch_norm_out, prelu_out

def replacement_args(concatenated_input, running_mean, running_var, weight, bias, prelu_weight):
    return (concatenated_input, running_mean, running_var, weight, bias, prelu_weight)

@triton.jit
def fused_batch_norm_prelu_kernel(
    input_ptr, output_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, alpha_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-3
):
    # Calculate total number of elements
    total_elements = N * C * H * W
    
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    
    # Calculate element offset
    element_offset = block_start + tl.arange(0, block_size)
    
    # Mask to ensure we don't go out of bounds
    mask = element_offset < total_elements
    
    # For 4D tensor [N, C, H, W], we need to calculate 2D index [N * H * W, C]
    # This is the preferred layout for better memory coalescing
    c_elements = H * W
    nhw_elements = N * c_elements
    
    # Calculate C and NHW indices
    c_indices = element_offset % C
    nhw_indices = element_offset // C
    
    # Load input data
    input_data = tl.load(input_ptr + element_offset, mask=mask, other=0.0)
    
    # Load batch norm parameters - each C channel gets its own params
    # Use tile to broadcast to all elements for each channel
    running_mean_data = tl.load(running_mean_ptr + c_indices, mask=mask, other=0.0)
    running_var_data = tl.load(running_var_ptr + c_indices, mask=mask, other=1.0)  # Use 1.0 as fallback for var
    weight_data = tl.load(weight_ptr + c_indices, mask=mask, other=1.0)          # Use 1.0 as fallback for weight
    bias_data = tl.load(bias_ptr + c_indices, mask=mask, other=0.0)              # Use 0.0 as fallback for bias
    alpha_data = tl.load(alpha_ptr + c_indices, mask=mask, other=0.0)           # Use 0.0 as fallback for alpha
    
    # Apply batch normalization
    # normalized = (input - mean) / sqrt(var + eps) * weight + bias
    normalized = (input_data - running_mean_data) / tl.sqrt(running_var_data + EPS) * weight_data + bias_data
    
    # Apply PReLU: max(0, x) * alpha for x < 0, otherwise just x
    # Note: PReLU uses alpha for negative values, zero for positive
    prelu_result = tl.where(normalized < 0.0, normalized * alpha_data, normalized)
    
    # Store the result
    tl.store(output_ptr + element_offset, prelu_result, mask=mask)

@torch.fx.wrap
def fused_batch_norm_prelu(input_tensor, running_mean, running_var, weight, bias, alpha):
    N, C, H, W = input_tensor.shape
    total_elements = N * C * H * W
    
    # Choose block size based on tensor size for optimal GPU occupancy
    if total_elements > 1024 * 1024:  # Large tensor
        BLOCK_SIZE = 4096
    elif total_elements > 1024:      # Medium tensor  
        BLOCK_SIZE = 2048
    else:                           # Small tensor
        BLOCK_SIZE = 1024
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch the kernel
    fused_batch_norm_prelu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        alpha_ptr=alpha,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    # Just return the fused function directly
    # The concat operation is handled by the pattern matching
    def wrapper(concatenated_input, running_mean, running_var, weight, bias, alpha):
        return fused_batch_norm_prelu(concatenated_input, running_mean, running_var, weight, bias, alpha)
    
    return wrapper