import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_0 = x.view(32, -1, 5, 32)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1

def replacement_args(x, y):
    return (x,)

@triton.jit
def optimized_view_transpose_5x32_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    feature_dim,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
):
    # Calculate program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1) * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    feat_idx = tl.program_id(2) * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    
    # Create masks for bounds checking
    batch_mask = batch_idx < batch_size
    seq_mask = seq_idx < seq_len
    feat_mask = feat_idx < feature_dim
    
    # Calculate input indices for [batch, seq, feature] -> [batch, seq//5, 5, feature//32, 32]
    # Then apply transpose(1,2) to get [batch, 5, seq//5, feature//32, 32]
    input_batch = batch_idx
    input_seq = seq_idx // 5  # seq dimension is divided by 5
    input_seq_remainder = seq_idx % 5  # remainder for the 5 dimension
    input_feat = feat_idx // 32  # feature dimension is divided by 32
    input_feat_remainder = feat_idx % 32  # remainder for the 32 dimension
    
    # Only process if we have valid indices
    mask = batch_mask & seq_mask & feat_mask
    # Also ensure we're within the bounds after division
    seq_valid = (seq_idx // 5) < (seq_len // 5)
    feat_valid = (feat_idx // 32) < (feature_dim // 32)
    mask = mask & seq_valid & feat_valid
    
    # Linearized input index
    input_indices = input_batch * seq_len * feature_dim + seq_idx * feature_dim + feat_idx
    
    # Load input data
    input_data = tl.load(input_ptr + input_indices, mask=mask, other=0.0)
    
    # After view and transpose: [batch, seq//5, 5, feature//32, 32] -> [batch, 5, seq//5, feature//32, 32]
    # We need to flatten the last two dimensions for the output tensor
    # The output will be [batch, 5, seq//5, (feature//32)*32] = [batch, 5, seq//5, feature]
    output_batch = batch_idx
    output_dim5 = input_seq_remainder  # This becomes the 5 dimension after transpose
    output_dim_seq_div5 = input_seq  # This becomes seq//5 dimension after transpose  
    output_feature = input_feat * 32 + input_feat_remainder  # Flatten the last two dimensions
    
    # Calculate output indices in final shape [batch, 5, seq//5, feature]
    output_indices = (output_batch * 5 + output_dim5) * (seq_len//5 * feature_dim) + output_dim_seq_div5 * feature_dim + output_feature
    
    # Store output
    tl.store(output_ptr + output_indices, input_data, mask=mask)

@torch.fx.wrap
def optimized_view_transpose_5x32(x):
    batch_size, seq_len, feature_dim = x.shape
    
    # For the specific pattern: [32, 1024, 160] -> [32, 5, 64, 160] after view and transpose
    # We need to handle the reshape correctly
    output_seq_div5 = seq_len // 5  # 1024 // 5 = 204.8... wait this doesn't make sense
    # Let me recalculate: this pattern seems different. Let me check the weight_meta again.
    # Actually, let me be more generic and handle this properly
    
    # The pattern: view(32, -1, 5, 32) means:
    # We want to split seq_len dimension into (-1, 5) and feature_dim dimension into (?, 32)
    # So: seq_len = -1 * 5, and feature_dim = ? * 32
    output_seq_div5 = seq_len // 5  # This should be an integer
    output_feat_div32 = feature_dim // 32  # This should be an integer
    
    # Calculate the actual output dimensions after view and transpose
    # [batch, seq_len, feature] -> view(batch, output_seq_div5, 5, output_feat_div32, 32) 
    # -> transpose(1,2) -> [batch, 5, output_seq_div5, output_feat_div32, 32]
    # -> flatten last two dims -> [batch, 5, output_seq_div5, feature]
    
    output = torch.empty((batch_size, 5, output_seq_div5, feature_dim), dtype=x.dtype, device=x.device)
    
    # Launch kernel with appropriate grid
    BLOCK_BATCH = 1
    BLOCK_SEQ = min(output_seq_div5, 256)
    BLOCK_FEAT = min(32, 64)
    
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (output_seq_div5 + BLOCK_SEQ - 1) // BLOCK_SEQ,
        (output_feat_div32 * 32 + BLOCK_FEAT - 1) // BLOCK_FEAT  # We process features in chunks of 32
    )
    
    optimized_view_transpose_5x32_kernel[grid](
        x,
        output,
        batch_size,
        seq_len,
        feature_dim
    )
    
    return output

def replacement_func():
    return optimized_view_transpose_5x32