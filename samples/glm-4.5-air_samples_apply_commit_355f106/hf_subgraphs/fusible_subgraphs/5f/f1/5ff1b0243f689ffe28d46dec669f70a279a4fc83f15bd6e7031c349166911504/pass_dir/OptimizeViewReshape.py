import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # The view operation pattern
    tmp_4 = in_1.view(-1, 512, -1)  # More flexible pattern to match both cases
    return (in_0, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def view_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_dim1,
    spatial_dim2,
    output_batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Create flattened view using Triton for optimal memory access
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset < (batch_size * channels * spatial_dim1 * spatial_dim2)
    
    # Load data with optimal coalescing
    input_ptrs = input_ptr + offset
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Store in output with same offset pattern (view preserves data layout)
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, x, mask=mask)

@torch.fx.wrap
def optimized_view_reshape(in_0, in_1):
    # Get original shape
    original_shape = in_1.shape
    batch_size, channels, spatial_dim1, spatial_dim2 = original_shape
    
    # Compute target shape: [batch_size, channels, spatial_dim1 * spatial_dim2]
    target_shape = (batch_size, channels, spatial_dim1 * spatial_dim2)
    
    # Create output tensor
    output = torch.empty(target_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Use only for larger tensors where Triton optimization makes sense
    total_elements = in_1.numel()
    if total_elements > 4096:  # Threshold for using kernel
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        view_reshape_kernel[(num_programs,)](
            in_1, output, batch_size, channels, spatial_dim1, spatial_dim2,
            BLOCK_SIZE
        )
    else:
        # For small tensors, use regular view (faster overhead)
        output = in_1.reshape(batch_size, channels, spatial_dim1 * spatial_dim2)
    
    return (in_0, output)

def replacement_func():
    return optimized_view_reshape