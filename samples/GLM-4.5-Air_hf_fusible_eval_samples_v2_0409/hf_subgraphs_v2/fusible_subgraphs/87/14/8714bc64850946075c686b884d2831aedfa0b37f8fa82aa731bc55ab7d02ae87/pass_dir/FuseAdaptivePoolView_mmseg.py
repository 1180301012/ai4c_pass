import torch
import triton
import triton.language as tl

# Pattern matching function - matching adaptive_avg_pool2d followed by view
def pattern(input_tensor, output_size, view_shape):
    """Match adaptive_avg_pool2d followed by view operation"""
    pooled = torch.nn.functional.adaptive_avg_pool2d(input_tensor, output_size)
    result = pooled.view(view_shape)
    return result, pooled  # Return both result and intermediate for observability

# Argument extraction function
def replacement_args(input_tensor, output_size, view_shape):
    return (input_tensor, output_size, view_shape)

# Optimized kernel that combines adaptive pooling and view operation
@triton.jit
def adaptive_pool_view_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    batch_size,
    input_height,
    input_width,
    view_shape_0,
    view_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (which corresponds to pooled output)
    pid = tl.program_id(0)
    
    # For 1x1 adaptive pooling, output is [batch_size, n_channels, 1, 1]
    # We'll compute the average over each channel across all spatial locations
    if pid < batch_size * n_channels:
        batch_idx = pid // n_channels
        channel_idx = pid % n_channels
        
        # Load and average all spatial elements for this channel and batch
        total = 0.0
        count = 0
        
        # Iterate over spatial dimensions
        for h in range(input_height):
            for w in range(input_width):
                input_offset = (batch_idx * n_channels + channel_idx) * input_height * input_width + h * input_width + w
                val = tl.load(input_ptr + input_offset, other=0.0)
                total += val
                count += 1
        
        if count > 0:
            avg_val = total / count
        else:
            avg_val = 0.0
        
        # Store in the first element of 1x1 spatial grid
        pooled_offset = (batch_idx * n_channels + channel_idx) * 1 * 1
        tl.store(output_ptr + pooled_offset, avg_val)
    
    # Handle view operation by simply reinterpreting the data
    # The kernel above already produces the correct shape for the final output

@torch.fx.wrap  
def optimized_adaptive_pool_view(input_tensor, output_size=1, view_shape=None):
    """
    Optimized function that fuses adaptive_avg_pool2d with size=1 and view operation
    """
    if output_size != 1:
        raise NotImplementedError("Only adaptive_avg_pool2d with size=1 is supported in this fusion")
    
    if view_shape is None:
        view_shape = (input_tensor.shape[0], input_tensor.shape[1])
    
    output = torch.empty(view_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    batch_size, n_channels, input_height, input_width = input_tensor.shape
    n_elements = batch_size * n_channels  # Output has 1x1 spatial dimensions
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    adaptive_pool_view_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_channels=n_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        view_shape_0=view_shape[0] if len(view_shape) > 0 else 1,
        view_shape_1=view_shape[1] if len(view_shape) > 1 else 1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference, not called directly)
def replacement_func():
    # Return a partially applied function with specific parameters
    def partial_apply(input_tensor, output_size=1, view_shape=None):
        return optimized_adaptive_pool_view(input_tensor, output_size, view_shape)
    return partial_apply