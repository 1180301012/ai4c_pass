import torch
import triton
import triton.language as tl
import math

@triton.jit
def linear_index_kernel(
    input_ptr,           # Pointer to input tensor [1, 15, 15, 512]
    weight_ptr,          # Pointer to weight tensor [features, 512]
    position_ptr,        # Pointer to position indices [64, 64]
    output_ptr,          # Pointer to output tensor [64, 64, features]
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1,
    position_stride_0, position_stride_1,
    output_stride_0, output_stride_1, output_stride_2,
    batch_size: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    in_channels: tl.constexpr,
    features: tl.constexpr,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles BLOCK_SIZE_M queries and BLOCK_SIZE_N features
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    # Compute bounds
    end_m = min(start_m + BLOCK_SIZE_M, num_queries)
    end_n = min(start_n + BLOCK_SIZE_N, features)
    
    # Process all queries in this block
    for m in range(start_m, end_m):
        # Load position index for current query
        pos_idx = tl.load(position_ptr + m * position_stride_0)
        
        # Process all features in this block
        for n in range(start_n, end_n):
            # Compute linear combination: input @ weight[n, :].t()
            acc = 0.0
            
            # Vectorized loading of weight for this feature
            weight = tl.load(weight_ptr + n * weight_stride_0, mask=n < features, other=0.0)
            
            # Load input at the specified position
            input_val = tl.load(input_ptr + pos_idx * input_stride_0, mask=pos_idx < (batch_size * in_height * in_width), other=0.0)
            
            # Compute dot product
            acc = tl.dot(input_val, weight)
            
            # Store result
            tl.store(output_ptr + m * output_stride_0 + n * output_stride_1, acc)

@torch.fx.wrap
def optimized_linear_index(input, weight, position_indices):
    # Get shapes
    batch_size, in_height, in_width, in_channels = input.shape
    features, _ = weight.shape
    
    # The position indices are [64, 64] and represent flattened positions
    num_queries = position_indices.numel()  # 64 * 64 = 4096
    
    # Output shape should be [num_queries, features]
    output_shape = (num_queries, features)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE_M = 64  # Number of queries per block
    BLOCK_SIZE_N = 64  # Number of features per block
    BLOCK_SIZE_K = 32  # Vectorization size
    
    num_blocks_m = (num_queries + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    linear_index_kernel[(num_blocks_m, num_blocks_n)](
        input, weight, position_indices, output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1),
        position_indices.stride(0), position_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        batch_size, in_height, in_width, in_channels,
        features, num_queries, num_queries // 64, 64,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def pattern(linear_input, weight_tensor, position_indices):
    """Pattern: linear transformation followed by indexing with position indices"""
    linear_result = torch.nn.functional.linear(linear_input, weight_tensor, None)
    viewed_result = linear_result.view(-1, weight_tensor.size(1)) if weight_tensor.dim() > 1 else linear_result.view(-1, weight_tensor.size(0))
    output = viewed_result[position_indices.view(-1)]
    
    # Reshape to match expected output [64, 64, features]
    features = weight_tensor.size(1) if weight_tensor.dim() > 1 else weight_tensor.size(0)
    output = output.view(64, 64, features)
    return output

def replacement_args(linear_input, weight_tensor, position_indices):
    return (linear_input, weight_tensor, position_indices)

def replacement_func():
    return optimized_linear_index