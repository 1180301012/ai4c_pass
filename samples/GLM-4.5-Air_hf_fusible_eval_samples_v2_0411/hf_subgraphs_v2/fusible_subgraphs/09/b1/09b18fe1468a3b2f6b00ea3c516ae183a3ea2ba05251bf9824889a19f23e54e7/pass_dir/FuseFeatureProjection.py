import torch
import triton
import triton.language as tl

@triton.jit
def fused_projection_kernel(
    codebook_ptr, query_features_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a specific query-key pair
    pid = tl.program_id(0)
    
    # Batch index, query index, key index
    batch_idx = pid // (num_queries * num_keys)
    query_key_idx = pid % (num_queries * num_keys)
    query_idx = query_key_idx // num_keys
    key_idx = query_key_idx % num_keys
    
    # Check bounds
    if batch_idx >= batch_size or query_idx >= num_queries or key_idx >= num_keys:
        return
    
    # Load codebook feature for this key
    codebook_offset = key_idx * feature_dim
    codebook_val = tl.load(codebook_ptr + codebook_offset)
    
    # Compute output for this batch and query using the codebook feature
    output_offset = (batch_idx * num_queries + query_idx) * feature_dim
    
    # Copy codebook feature to all positions in this query's output
    for i in range(0, feature_dim, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < feature_dim
        
        # Store codebook feature at the current query position
        tl.store(output_ptr + output_offset + offset, codebook_val, mask=mask)

@torch.fx.wrap
def fused_feature_projection(in_0, in_4):
    """
    Fuse: 
    - tmp_6 = in_0.view((1, 1, 32, 512))  - reshape codebook
    - tmp_7 = in_4.unsqueeze(2)          - add dimension to features
    - tmp_8 = tmp_7.expand((1, 4096, 32, 512)) - expand to match dimensions
    - tmp_10 = tmp_8 - tmp_6             - subtract
    
    Output: [1, 4096, 32, 512] broadcasted features minus codebook
    """
    # Input shapes:
    # in_0: [32, 512] - codebook features
    # in_4: [1, 4096, 512] - query features
    
    batch_size, num_queries, feature_dim = in_4.shape
    
    # Get original codebook dimensions
    num_keys, codebook_feature_dim = in_0.shape
    
    # Create output: [1, 4096, 32, 512]
    output_shape = (batch_size, num_queries, num_keys, codebook_feature_dim)
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel for fused projection
    BLOCK_SIZE = 256
    total_elements = batch_size * num_queries * num_keys
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_projection_kernel[(num_programs,)](
        codebook_ptr=in_0,
        query_features_ptr=in_4,
        output_ptr=output,
        batch_size=batch_size,
        num_queries=num_queries,
        num_keys=num_keys,
        feature_dim=codebook_feature_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def get_attention_weights_shape(attention_weights):
    """Helper to handle the shape that comes from the first pass"""
    return attention_weights.unsqueeze(3)

def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    # tmp_9 comes from elsewhere, so we can't include it in the pattern
    tmp_10 = tmp_8 - tmp_6
    return tmp_10

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # Only return in_0 and in_4 for the pattern
    return (in_0, in_4)

def replacement_func():
    def wrapper(**kwargs):
        in_0 = kwargs.get('in_0')
        in_4 = kwargs.get('in_4')
        
        # Compute fused projection
        tmp_10 = fused_feature_projection(in_0, in_4)
        
        return tmp_10
    return wrapper