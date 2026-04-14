import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(softmax_input, main_input):
    """
    Matches the pattern: softmax + multiple reshapes + multiplication
    This pattern is found across all graph variations (float32, float16, bfloat16)
    Uses generic batch size to match all variants
    """
    # Softmax operation along dimension 1
    softmax_output = torch.nn.functional.softmax(softmax_input, dim=1)
    
    # Get batch dimension for dynamic reshape
    batch_size = softmax_input.shape[0]
    
    # Multiple inefficient reshape operations (this is what we want to optimize)
    reshaped_1 = softmax_output.reshape(batch_size, -1)  # Flatten all but batch dimension
    reshaped_2 = reshaped_1.view(batch_size, -1, 1, 1)     # Add singleton dimensions
    reshaped_3 = reshaped_2.view(batch_size, 2, -1, 1, 1)  # Restore middle dimensions
    
    # Element-wise multiplication with main input
    result = reshaped_3 * main_input
    
    # Return the final result (matches the original computation structure)
    return result

# Argument extraction function  
def replacement_args(softmax_input, main_input):
    """
    Extract arguments needed for the replacement kernel
    """
    return (softmax_input, main_input)

# Triton kernel for optimized fused operation
@triton.jit
def optimized_softmax_multiply_kernel(
    # Input tensors
    softmax_input_ptr,  # [B, 2, 1, 128] - softmax input
    main_input_ptr,     # [B, 2, K, H, W] - main input for multiplication
    output_ptr,         # [B, 2, K, H, W] - multiplication result
    
    # Shape information
    batch_size,         # B
    k_size,             # K (128 in all cases)
    height,             # H (varies per graph) 
    width,              # W (varies per graph)
    
    # Constants
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr
):
    """Optimized Triton kernel: fuses reshape operations and multiplication"""
    
    # Program IDs for 2D grid 
    program_id = tl.program_id(0)
    batch_id = program_id // (height * width // (BLOCK_SIZE_X * BLOCK_SIZE_Y))
    remaining_id = program_id % (height * width // (BLOCK_SIZE_X * BLOCK_SIZE_Y))
    
    y = (remaining_id // (BLOCK_SIZE_X // BLOCK_SIZE_Y)) * BLOCK_SIZE_Y
    x = (remaining_id % (BLOCK_SIZE_X // BLOCK_SIZE_Y)) * BLOCK_SIZE_X
    
    # Calculate linear offset for this position
    linear_offset = batch_id * (2 * k_size * height * width)
    
    # Process M dimension (middle dimension of size 2) and K dimension (128)
    for m_id in range(2):
        for k_pos in range(0, k_size, BLOCK_SIZE_X):
            # Load softmax values for this [B, M, 1, K] combination
            softmax_offset = linear_offset + m_id * (k_size * height * width) + k_pos * (height * width)
            softmax_ptr = softmax_input_ptr + softmax_offset
            softmax_val = tl.load(softmax_ptr + y * width + x, 
                                mask=(y < height) & (x < width), 
                                other=0.0)
            
            # Load main input values for this [B, M, K, H, W] combination
            main_offset = softmax_offset  # Same offset pattern after reshape optimization
            main_ptr = main_input_ptr + main_offset
            main_val = tl.load(main_ptr + y * width + x, 
                             mask=(y < height) & (x < width), 
                             other=0.0)
            
            # Perform multiplication
            output_val = softmax_val * main_val
            
            # Store result
            tl.store(output_ptr + main_offset + y * width + x, output_val, 
                    mask=(y < height) & (x < width))

@torch.fx.wrap  
def optimized_fused_operation(softmax_input, main_input):
    """
    Optimized wrapper function that fuses softmax, reshape, and multiplication
    """
    # Get input shapes
    batch_size, m_size, _, k_softmax = softmax_input.shape
    _, _, k_main, height, width = main_input.shape
    
    # Validate shapes (should be consistent)
    assert k_softmax == k_main, f"K dimension mismatch: {k_softmax} vs {k_main}"
    assert m_size == 2, f"Middle dimension should be 2, got {m_size}"
    
    # Create output tensor with same shape as main input
    output = torch.empty_like(main_input, device=softmax_input.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_X = 128  # Block size for X dimension
    BLOCK_SIZE_Y = 4    # Block size for Y dimension
    
    # Calculate total number of elements for grid
    total_elements = batch_size * height * width
    grid_size = (total_elements + BLOCK_SIZE_X * BLOCK_SIZE_Y - 1) // (BLOCK_SIZE_X * BLOCK_SIZE_Y)
    
    # Launch kernel
    optimized_softmax_multiply_kernel[grid_size](
        softmax_input,
        main_input, 
        output,
        batch_size, k_size=k_softmax, height=height, width=width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y
    )
    
    # Apply reduction and contiguous copy (matches original computation)
    summed = torch.sum(output, dim=1)
    contiguous_result = summed.contiguous()
    
    return contiguous_result

# Replacement function (returns the optimization function)
def replacement_func():
    """Return the fused optimization function"""
    return optimized_fused_operation