import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation structure
def pattern(in_0, in_1, in_2):
    # Match the complete computation graph
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = torch.empty_like(tmp_1)  # Will fill with zeros in optimized version
    return (tmp_3, tmp_4, tmp_1)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    # Extract all necessary arguments
    return (in_0, in_1, in_2, in_1.shape[0], in_2.shape[1])

# Optimized Triton kernel for the entire computation
@triton.jit
def optimized_compute_kernel(
    edge_index_ptr,
    edge_weight_ptr,
    feature_ptr,
    output_3_ptr,
    output_4_ptr,
    output_1_ptr,
    edge_size,
    feature_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < edge_size
    
    # Load edge weights (for output_1 - broadcast multiplication result)
    edge_weight = tl.load(edge_weight_ptr + idx, mask=mask, other=0.0)
    
    # Load features - each element gets the entire row of features
    feature_vals = tl.load(feature_ptr + idx * feature_size + tl.arange(0, feature_size), 
                          mask=idx < edge_size, other=0.0)
    
    # Load edge indices (for output_3 - expanded version)
    edge_index = tl.load(edge_index_ptr + idx, mask=mask, other=0.0)
    
    # Compute output_1: edge_weight * feature_row (broadcast multiplication)
    # Since this is element-wise multiplication per feature row
    output_1 = edge_weight * feature_vals
    
    # Compute output_3: expand edge_index to match feature dimensions
    # Each edge_index value is repeated for all features
    output_3 = tl.broadcast_to(edge_index, feature_size)
    
    # Store output_1 (broadcast multiplication result)
    tl.store(output_1_ptr + idx * feature_size + tl.arange(0, feature_size), 
             output_1, mask=idx < edge_size)
    
    # Store output_3 (expanded edge indices)
    tl.store(output_3_ptr + idx * feature_size + tl.arange(0, feature_size), 
             output_3, mask=idx < edge_size)
    
    # Store output_4: zeros tensor (same shape as output_1)
    tl.store(output_4_ptr + idx * feature_size + tl.arange(0, feature_size), 
             0.0, mask=idx < edge_size)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_full_compute(in_0, in_1, in_2, edge_size, feature_size):
    BLOCK_SIZE = 1024
    total_elements = edge_size * feature_size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output shapes: edge_size x feature_size for all outputs
    output_shape = (edge_size, feature_size)
    
    # Create output tensors
    output_3 = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    output_4 = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    output_1 = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Optimized kernel that computes all outputs in one pass
    optimized_compute_kernel[(num_programs,)](
        edge_index_ptr=in_0,
        edge_weight_ptr=in_1,
        feature_ptr=in_2,
        output_3_ptr=output_3,
        output_4_ptr=output_4,
        output_1_ptr=output_1,
        edge_size=edge_size,
        feature_size=feature_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (output_3, output_4, output_1)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_full_compute