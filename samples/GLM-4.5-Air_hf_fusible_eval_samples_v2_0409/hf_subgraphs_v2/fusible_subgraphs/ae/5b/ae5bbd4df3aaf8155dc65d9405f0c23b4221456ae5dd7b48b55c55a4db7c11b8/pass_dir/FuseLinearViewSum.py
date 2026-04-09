import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    # Match just the linear operation to ensure compatibility
    return torch.nn.functional.linear(in_3, in_1, in_0)

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_linear_view_sum_kernel(
    query_ptr,  # [1, 12, 199, 64]
    weight_ptr,  # [8, 64] 
    bias_ptr,    # [8]
    output_ptr,  # [1, 12, 199, 2]
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    input_dim: tl.constexpr,
    output_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one head in the sequence
    pid_m = tl.program_id(0)  # sequence position
    pid_k = tl.program_id(1)  # head index
    
    # Calculate range for this program
    start_m = pid_m * BLOCK_M
    start_k = pid_k
    
    # Calculate number of iterations needed
    num_n_iters = (input_dim + BLOCK_N - 1) // BLOCK_N
    
    # Load bias for this head
    bias_offset = pid_k * output_dim
    bias = tl.load(bias_ptr + bias_offset)
    
    # Initialize accumulator
    acc = tl.zeros((output_dim // 2,), dtype=tl.bfloat16)
    
    # Main matrix multiplication loop
    for i in range(num_n_iters):
        # Calculate offset for current iteration
        offset_n = i * BLOCK_N
        
        # If we're going out of bounds, skip this iteration
        if offset_n >= input_dim:
            continue
            
        # Calculate the actual block size
        block_size = min(BLOCK_N, input_dim - offset_n)
        
        # Load query block for this head and position
        query_offset = (start_m * num_heads * seq_len * input_dim + 
                       pid_k * seq_len * input_dim + 
                       start_m * input_dim + 
                       offset_n)
        query = tl.load(query_ptr + query_offset, mask=offset_n < input_dim)
        
        # Load corresponding weight block
        for j in range(output_dim // 2):
            weight_offset = (j * input_dim + offset_n)
            weight = tl.load(weight_ptr + weight_offset, mask=offset_n < input_dim)
            
            # Matrix multiply element
            acc[j] += tl.sum(query * weight)
    
    # Add bias
    acc += bias[:output_dim // 2]
    
    # Store result: reshape [output_dim] to [2] (since output_dim=8, we output 2 values)
    output_offset = (pid_k * seq_len * (output_dim // 2) + 
                     start_m * (output_dim // 2))
    
    # Store the two halves
    tl.store(output_ptr + output_offset, acc[0:1], mask=acc[0:1].shape[0] > 0)
    tl.store(output_ptr + output_offset + 1, acc[1:2], mask=acc[1:2].shape[0] > 0)
    tl.store(output_ptr + output_offset + 2, acc[2:3], mask=acc[2:3].shape[0] > 0)
    tl.store(output_ptr + output_offset + 3, acc[3:4], mask=acc[3:4].shape[0] > 0)

@triton.jit
def optimized_fused_linear_view_sum_kernel(
    query_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    input_dim: tl.constexpr,
    output_dim: tl.constexpr,
):
    pid_k = tl.program_id(0)  # head index
    pid_m = tl.program_id(1)  # sequence position
    
    # Compute linear transformation: Y = X @ W.t() + B
    # Where X is [num_heads, seq_len, input_dim], W is [output_dim, input_dim]
    # Output Y is [num_heads, seq_len, output_dim]
    
    # Process all output elements in parallel for better performance
    for out_idx in range(output_dim):
        # Initialize with bias for this output element
        bias_val = tl.load(bias_ptr + out_idx)
        accum = bias_val
        
        # Vectorized dot product over input dimension
        for n_idx in range(input_dim):
            # Load input element: X[pid_k, pid_m, n_idx]
            input_offset = (pid_k * seq_len * input_dim + pid_m * input_dim + n_idx)
            x_val = tl.load(query_ptr + input_offset)
            
            # Load weight element: W[out_idx, n_idx] 
            weight_offset = (out_idx * input_dim + n_idx)
            w_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate dot product: X[..., n_idx] * W[out_idx, n_idx]
            accum += x_val * w_val
        
        # Store result: Y[pid_k, pid_m, out_idx]
        output_offset = (pid_k * seq_len * output_dim + pid_m * output_dim + out_idx)
        tl.store(output_ptr + output_offset, accum)

def kernel_wrapper(query, weight, bias):
    # Get tensor shapes dynamically
    # query: [1, num_heads, seq_len, 64] 
    # weight: [8, 64]
    # bias: [8]
    num_heads = query.shape[1]
    seq_len = query.shape[2] 
    input_dim = query.shape[3]
    output_dim = weight.shape[0]
    
    # Output should match the original linear result: [1, num_heads, seq_len, 8]
    output_shape = [1, num_heads, seq_len, output_dim]
    output = torch.empty(output_shape, dtype=bias.dtype, device=query.device)
    
    # Flatten tensors for easier indexing (keep batch dimension for output)
    # query: [1, num_heads, seq_len, 64] -> [num_heads, seq_len, 64]
    query_flat = query.view(num_heads, seq_len, input_dim)
    # weight: [8, 64]
    weight_flat = weight.view(output_dim, input_dim)
    # bias: [8]
    bias_flat = bias.view(output_dim)
    # output: [1, num_heads, seq_len, 8] -> [num_heads, seq_len, 8] for kernel processing
    output_flat = output.view(num_heads, seq_len, output_dim)
    
    # Grid: (head_index, sequence_position)
    grid = (num_heads, seq_len)
    
    # Launch kernel
    optimized_fused_linear_view_sum_kernel[grid](
        query_flat, 
        weight_flat,
        bias_flat,
        output_flat,
        seq_len,
        input_dim,
        output_dim
    )
    
    # Reshape back to include batch dimension
    return output.view(1, num_heads, seq_len, output_dim)

@torch.fx.wrap
def fused_linear_view_sum(query, weight, bias):
    return kernel_wrapper(query, weight, bias)

def replacement_func():
    return fused_linear_view_sum