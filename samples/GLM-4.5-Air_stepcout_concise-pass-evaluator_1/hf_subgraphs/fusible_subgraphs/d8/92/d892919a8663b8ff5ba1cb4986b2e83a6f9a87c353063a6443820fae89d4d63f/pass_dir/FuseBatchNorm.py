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
    grid_0, grid_1, grid_2,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    EPS: tl.constexpr = 1e-3
):
    # Calculate total number of elements
    total_elements = N * C * H * W
    
    # Get program ID - use full 3D grid for better GPU utilization
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1) 
    pid_2 = tl.program_id(2)
    
    # Create unique program ID from 3D grid
    pid = pid_0 * grid_1 * grid_2 + pid_1 * grid_2 + pid_2
    
    # Calculate this program's work range
    num_programs = grid_0 * grid_1 * grid_2
    elements_per_program = (total_elements + num_programs - 1) // num_programs
    block_start = pid * elements_per_program
    
    # Calculate work range for this program
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # For 4D tensors [N, C, H, W], calculate channel index for each element
    # Each channel has its own parameters that apply to all HxW spatial positions
    c_indices = offsets % C
    
    # Load parameters - each channel gets its own mean/var/weight/bias
    # These will be broadcast to all spatial positions for that channel
    running_mean_data = tl.load(running_mean_ptr + c_indices, mask=mask, other=0.0)
    running_var_data = tl.load(running_var_ptr + c_indices, mask=mask, other=1.0)
    weight_data = tl.load(weight_ptr + c_indices, mask=mask, other=1.0)
    bias_data = tl.load(bias_ptr + c_indices, mask=mask, other=0.0)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (input_data - running_mean_data) / tl.sqrt(running_var_data + EPS) * weight_data + bias_data
    
    # Store the result
    tl.store(output_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    # Ensure input tensor is contiguous for correct memory layout
    input_tensor = input_tensor.contiguous()
    
    N, C, H, W = input_tensor.shape
    total_elements = N * C * H * W
    
    # Use moderate block size to balance occupancy and register usage
    BLOCK_SIZE = 1024
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor and ensure it's contiguous too
    output = torch.empty_like(input_tensor, dtype=input_tensor.dtype)
    
    # Launch the kernel with simpler grid configuration
    optimized_batch_norm_kernel[(num_programs, 1, 1)](
        input_ptr=input_tensor,
        output_ptr=output,
        running_mean_ptr=running_mean.contiguous() if running_mean is not None else running_mean,
        running_var_ptr=running_var.contiguous() if running_var is not None else running_var,
        weight_ptr=weight.contiguous() if weight is not None else weight,
        bias_ptr=bias.contiguous() if bias is not None else bias,
        grid_0=num_programs, grid_1=1, grid_2=1,
        BLOCK_SIZE=BLOCK_SIZE,
        N=N, C=C, H=H, W=W
    )
    
    return output

def replacement_func():
    def wrapper(input_tensor, running_mean, running_var, weight, bias):
        return optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias)
    
    return wrapper