import torch
import triton
import triton.language as tl

@triton.jit
def view_permute_kernel(
    input_ptr,           # Pointer to input tensor [num_queries, features]
    output_ptr,          # Pointer to output tensor [features, height, width]
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1, output_stride_2,
    height: tl.constexpr,
    width: tl.constexpr,
    features: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles BLOCK_SIZE_M features and BLOCK_SIZE_N spatial locations
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    # Compute bounds
    end_m = min(start_m + BLOCK_SIZE_M, features)
    end_n = min(start_n + BLOCK_SIZE_N, height * width)
    
    # Process features in this block
    for m in range(start_m, end_m):
        # Process spatial locations in this block
        for n in range(start_n, end_n):
            # Decompose spatial location into h, w coordinates
            h_idx = n // width
            w_idx = n % width
            
            # Input index: [n, m] where n is flattened spatial location, m is feature
            input_idx = n * input_stride_0 + m * input_stride_1
            
            # Output index: [m, h_idx, w_idx]
            output_idx = m * output_stride_0 + h_idx * output_stride_1 + w_idx * output_stride_2
            
            # Load from input, store to output
            val = tl.load(input_ptr + input_idx)
            tl.store(output_ptr + output_idx, val)

@torch.fx.wrap
def optimized_view_permute(input_tensor, target_shape):
    """
    Optimized version that directly reshapes from [num_queries, features] to [features, height, width]
    without intermediate permute and contiguous operations.
    """
    num_queries, features = input_tensor.shape
    height, width = target_shape
    
    # Validate shapes
    assert num_queries == height * width, f"Expected {height * width} queries, got {num_queries}"
    
    # Create output tensor directly in target shape [features, height, width]
    output = torch.empty((features, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE_M = 64  # Number of features per block
    BLOCK_SIZE_N = 64  # Number of spatial locations per block
    
    num_blocks_m = (features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    view_permute_kernel[(num_blocks_m, num_blocks_n)](
        input_tensor, output,
        input_tensor.stride(0), input_tensor.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        height, width, features,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def pattern(input_tensor, target_shape=(64, 64)):
    """
    Pattern: view(64, 64, -1) -> permute(2, 0, 1) -> contiguous()
    This reshapes from [64, 64, features] to [features, 64, 64]
    """
    height, width = target_shape
    
    # The input should be [64, 64, features] based on the view operation
    reshaped = input_tensor  # This should already be in [64, 64, features] shape
    
    # Apply permute and contiguous
    permuted = reshaped.permute(2, 0, 1)     # [features, 64, 64]
    result = permuted.contiguous()           # Same shape, guaranteed contiguous
    
    return result

def replacement_args(input_tensor, target_shape=(64, 64)):
    return (input_tensor, target_shape)

def replacement_func():
    return optimized_view_permute