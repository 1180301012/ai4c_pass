import torch
import triton
import triton.language as tl

def pattern(input_tensor, pool_size):
    # Adaptive average pooling
    pool_out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, pool_size)
    
    # Reshape operation - this can be fused/optimized
    view_out = pool_out.view(-1, input_tensor.shape[1])
    
    return pool_out, view_out

def replacement_args(input_tensor, pool_size):
    return (input_tensor, pool_size)

@triton.jit
def adaptive_pool2d_view_kernel(
    input_ptr, output_ptr,
    N, C, H, W, output_H, output_W,
    BLOCK_SIZE: tl.constexpr
):
    # For adaptive_avg_pool2d with output_size=1, each output pixel corresponds
    # to the average of the entire input region (the whole spatial dimension)
    # So for output_size=1, we're essentially computing the spatial mean
    
    total_output_elements = N * C * output_H * output_W
    total_input_elements = N * C * H * W
    
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    
    # Calculate element offset for output
    output_offset = block_start + tl.arange(0, block_size)
    mask = output_offset < total_output_elements
    
    # For output_size=1, each output element [n, c, 0, 0] corresponds to
    # the average of all elements [n, c, h, w] for all h, w
    # So we need to compute: sum(input_n_c_h_w) / (H * W) for each n, c
    
    # Calculate indices for output [N, C, 1, 1] -> flatten to [N*C, 1, 1]
    output_c_indices = output_offset % C
    output_nc_indices = output_offset // C
    
    # For each output element, we need to sum all corresponding input elements
    # Input is [N, C, H, W], we access elements [n, c, h, w] for each h, w
    
    # The number of input elements per output element
    elements_per_output = H * W
    
    # Initialize accumulator for sum
    result_sum = 0.0
    
    # Loop over all input elements that contribute to this output
    # This is better done on GPU by having each thread compute one output element
    # and launch enough threads to cover all outputs
    
    # For each output element, compute the mean of all input elements for that channel
    # Since output_size=1, we process each output element independently
    
    # Load the entire channel and compute mean using efficient reduction
    # This is a simplified approach - for better performance, we'd want
    # to do a more sophisticated parallel reduction
    
    # For now, let's use a simple approach that works well for adaptive_avg_pool2d(1)
    # Each thread computes the mean for one output element
    
    # Convert to 2D indexing: [N*H*W, C]
    input_c_elements = W
    input_nhw_elements = N * H * input_c_elements
    
    # For this output element, compute mean over all h, w for given n, c
    n = output_nc_indices // output_H  # N index (for output size 1, output_H=1)
    c = output_c_indices
    
    # Initialize sum for this output element
    channel_sum = 0.0
    
    # Sum over all spatial positions (h, w)
    for h in range(H):
        for w in range(W):
            # Calculate input index for [n, c, h, w]
            input_offset_3d = n * H * W + h * W + w  # Flattened spatial first
            input_offset = input_offset_3d * C + c    # Add channel
            
            # Make sure we don't go out of bounds
            if input_offset < total_input_elements:
                input_val = tl.load(input_ptr + input_offset, other=0.0)
                channel_sum += input_val
    
    # Compute mean: divide by number of spatial elements
    mean_val = channel_sum / (H * W)
    
    # Store the result
    tl.store(output_ptr + output_offset, mean_val, mask=mask)

@torch.fx.wrap  
def optimized_adaptive_pool2d_view(input_tensor, output_size):
    N, C, H, W = input_tensor.shape
    output_H, output_W = output_size
    total_output_elements = N * C * output_H * output_W
    
    # Handle the specific case where output_size=(1,1) which is very common for our use case
    if output_size == (1, 1):
        # More optimized path for output_size=(1,1)
        total_elements = total_output_elements
        
        if total_elements > 1024 * 1024:
            BLOCK_SIZE = 4096
        elif total_elements > 1024:
            BLOCK_SIZE = 2048
        else:
            BLOCK_SIZE = 1024
        
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Create output tensor [N, C, 1, 1]
        output_pool = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Launch kernel for pooling - this computes spatial mean directly
        adaptive_pool2d_view_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output_pool.view(-1),  # Flatten for easier 1D indexing
            N=N, C=C, H=H, W=W, output_H=output_H, output_W=output_W,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Apply the view operation - this is cheap and can be done directly
        output_view = output_pool.view(N, C)
        
        return output_pool, output_view
    else:
        # This path should not be reached since we only match output_size=1
        # But keep a simple fallback for safety
        raise NotImplementedError("Only output_size=(1,1) is supported by this optimization pass")

def replacement_func():
    def wrapper(input_tensor, pool_size):
        return optimized_adaptive_pool2d_view(input_tensor, pool_size)
    
    return wrapper