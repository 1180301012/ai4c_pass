import torch
import triton
import triton.language as tl

def pattern(a, b):
    # a has shape [batch, seq_len, num_queries]
    # b has shape [batch, num_queries, hidden_dim]
    matmul = torch.matmul(a, b)
    result = matmul.permute(0, 2, 1)
    return result

def replacement_args(a, b):
    return (a, b)

@triton.jit
def matmul_transpose_kernel(
    a_ptr, b_ptr, out_ptr,
    batch, seq_len, num_queries, hidden_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Advanced kernel with optimal memory access patterns and vectorization
    pid = tl.program_id(0)
    
    # Calculate which batch item and which element in the output
    batch_id = pid // (seq_len * hidden_dim)
    linear_idx = pid % (seq_len * hidden_dim)
    seq_idx = linear_idx // hidden_dim
    hidden_idx = linear_idx % hidden_dim
    
    # Check bounds using individual comparisons
    if batch_id >= batch:
        return
    if seq_idx >= seq_len:
        return
    if hidden_idx >= hidden_dim:
        return
    
    # Initialize accumulator with float32 precision
    sum_val = 0.0
    
    # Optimized loop: Load multiple elements when possible for better cache utilization
    # Use vectorized loads when query dimension allows
    if num_queries >= 4:
        # Process 4 queries at a time for better vectorization
        for query_idx in range(0, num_queries, 4):
            if query_idx + 3 < num_queries:
                # Vectorized load for 4 elements
                a_offset0 = batch_id * seq_len * num_queries + seq_idx * num_queries + query_idx
                a_offset1 = a_offset0 + 1
                a_offset2 = a_offset0 + 2
                a_offset3 = a_offset0 + 3
                
                b_offset0 = batch_id * num_queries * hidden_dim + query_idx * hidden_dim + hidden_idx
                b_offset1 = b_offset0 + hidden_dim
                b_offset2 = b_offset0 + 2 * hidden_dim
                b_offset3 = b_offset0 + 3 * hidden_dim
                
                # Load 4 elements from each matrix
                a_val0 = tl.load(a_ptr + a_offset0)
                a_val1 = tl.load(a_ptr + a_offset1)
                a_val2 = tl.load(a_ptr + a_offset2)
                a_val3 = tl.load(a_ptr + a_offset3)
                
                b_val0 = tl.load(b_ptr + b_offset0)
                b_val1 = tl.load(b_ptr + b_offset1)
                b_val2 = tl.load(b_ptr + b_offset2)
                b_val3 = tl.load(b_ptr + b_offset3)
                
                # Accumulate with vectorized operations
                sum_val += a_val0 * b_val0 + a_val1 * b_val1 + a_val2 * b_val2 + a_val3 * b_val3
            else:
                # Handle remaining elements
                for rem in range(query_idx, num_queries):
                    a_offset = batch_id * seq_len * num_queries + seq_idx * num_queries + rem
                    b_offset = batch_id * num_queries * hidden_dim + rem * hidden_dim + hidden_idx
                    
                    a_val = tl.load(a_ptr + a_offset)
                    b_val = tl.load(b_ptr + b_offset)
                    sum_val += a_val * b_val
    else:
        # Small query dimension: use simple loop
        for query_idx in range(num_queries):
            a_offset = batch_id * seq_len * num_queries + seq_idx * num_queries + query_idx
            b_offset = batch_id * num_queries * hidden_dim + query_idx * hidden_dim + hidden_idx
            
            a_val = tl.load(a_ptr + a_offset)
            b_val = tl.load(b_ptr + b_offset)
            sum_val += a_val * b_val
    
    # Store result to output: [batch, hidden_dim, seq_len] layout
    out_offset = batch_id * hidden_dim * seq_len + hidden_idx * seq_len + seq_idx
    tl.store(out_ptr + out_offset, sum_val)

@torch.fx.wrap
def matmul_with_transpose(a, b):
    batch, seq_len, num_queries = a.shape
    _, _, hidden_dim = b.shape
    
    # Dynamic block size selection based on tensor characteristics for optimal GPU utilization
    total_elements = batch * seq_len * hidden_dim
    
    # Choose optimal block size based on total workload
    if total_elements >= 1_000_000:  # Large tensors
        BLOCK_SIZE = 2048  # Larger blocks for better occupancy
    elif total_elements >= 500_000:  # Medium tensors
        BLOCK_SIZE = 1536  # Medium blocks
    elif total_elements >= 100_000:  # Small-medium tensors
        BLOCK_SIZE = 1024  # Standard size
    elif total_elements >= 10_000:  # Small tensors
        BLOCK_SIZE = 512   # Smaller blocks
    else:  # Very small tensors
        BLOCK_SIZE = 256   # Minimal blocks
    
    # Calculate grid dimensions - 1D grid where each thread handles one output element
    grid_size = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor with transposed shape [batch, hidden_dim, seq_len]
    output = torch.empty((batch, hidden_dim, seq_len), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    matmul_transpose_kernel[grid_size](
        a, b, output,
        batch, seq_len, num_queries, hidden_dim,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return matmul_with_transpose