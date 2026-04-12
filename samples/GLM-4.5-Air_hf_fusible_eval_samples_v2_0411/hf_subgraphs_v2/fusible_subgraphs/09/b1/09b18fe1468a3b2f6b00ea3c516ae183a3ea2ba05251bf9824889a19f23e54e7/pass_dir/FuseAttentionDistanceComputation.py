import torch
import triton
import triton.language as tl

@triton.jit
def squared_difference_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    batch_size: tl.constexpr,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of the output
    pid = tl.program_id(0)
    
    # Get batch, query, and key indices for this program
    batch_idx = pid // (num_queries * num_keys)
    query_key_idx = pid % (num_queries * num_keys)
    query_idx = query_key_idx // num_keys
    key_idx = query_key_idx % num_keys
    
    # Check bounds
    if batch_idx >= batch_size or query_idx >= num_queries or key_idx >= num_keys:
        return
    
    # Initialize sum for this query-key pair with higher precision
    sum_val = 0.0
    
    # Optimized loop: process larger chunks to reduce TLB misses
    for i in range(0, feature_dim, BLOCK_SIZE * 4):
        # Process multiple elements at once
        offsets = i + tl.arange(0, BLOCK_SIZE * 4)
        mask = offsets < feature_dim
        
        # Load blocks with better memory coalescing
        x = tl.load(x_ptr + batch_idx * num_queries * num_keys * feature_dim + 
                   query_idx * num_keys * feature_dim + key_idx * feature_dim + offsets,
                   mask=mask, other=0.0)
        y = tl.load(y_ptr + batch_idx * num_queries * num_keys * feature_dim + 
                   query_idx * num_keys * feature_dim + key_idx * feature_dim + offsets,
                   mask=mask, other=0.0)
        
        # Compute squared differences and accumulate
        diff = x - y
        sum_val += tl.sum(diff * diff)
    
    # Store the final sum
    out_offset = batch_idx * num_queries * num_keys + query_idx * num_keys + key_idx
    tl.store(out_ptr + out_offset, sum_val)

@torch.fx.wrap
def fused_squared_difference_sum(in_1, in_2):
    # Implement: tmp_1 = in_1 - in_2, tmp_2 = tmp_1.pow(2), tmp_3 = tmp_2.sum(dim=3)
    
    batch_size, num_queries, num_keys, feature_dim = in_1.shape
    
    # Create output for sums: [batch_size, num_queries, num_keys]
    output_shape = (batch_size, num_queries, num_keys)
    tmp_3 = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel for fused computation
    BLOCK_SIZE = 256
    total_elements = batch_size * num_queries * num_keys
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    squared_difference_kernel[(num_programs,)](
        x_ptr=in_1,
        y_ptr=in_2,
        out_ptr=tmp_3,
        batch_size=batch_size,
        num_queries=num_queries,
        num_keys=num_keys,
        feature_dim=feature_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return tmp_3

def pattern(in_1, in_2):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    return tmp_3

def replacement_args(*args):
    # Handle flexible argument passing - extract in_1 and in_2
    # args could be in_0, in_1, in_2, in_3, in_4 or other combinations
    if len(args) >= 3:
        return (args[1], args[2])  # in_1, in_2
    elif len(args) == 2:
        return args  # in_1, in_2
    else:
        raise ValueError(f"Unexpected number of arguments: {len(args)}")

def replacement_func():
    return fused_squared_difference_sum