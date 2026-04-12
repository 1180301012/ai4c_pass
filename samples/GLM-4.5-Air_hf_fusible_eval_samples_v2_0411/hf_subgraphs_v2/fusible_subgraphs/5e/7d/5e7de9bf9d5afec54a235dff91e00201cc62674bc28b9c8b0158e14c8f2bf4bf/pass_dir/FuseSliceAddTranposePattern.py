import torch
import triton
import triton.language as tl

def pattern(tmp_5, tmp_4):
    # Slice operations: both tensors are sliced to the same shape [1, 768, 124]
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    
    # Element-wise addition
    tmp_8 = tmp_6 + tmp_7
    
    # Transpose operation
    tmp_9 = tmp_8.transpose(1, 2)
    
    return tmp_9

def replacement_args(tmp_5, tmp_4):
    return (tmp_5, tmp_4)

@triton.jit
def vectorized_add_transpose_kernel(
    input1_ptr,     # tmp_5: [1, features, seq1]
    input2_ptr,     # tmp_4: [1, features, seq2]
    output_ptr,     # [1, seq_out, features] - transposed result  
    n_features,     # Number of features (768)
    n_seq_out: tl.constexpr,  # Output sequence length (124)
    n_seq1,         # Input1 sequence length (125)
    n_seq2,         # Input2 sequence length (235)
    block_size: tl.constexpr,
):
    # Program handles a block of features
    feat_start = tl.program_id(0) * block_size
    feat_end = min(feat_start + block_size, n_features)
    
    if feat_start >= n_features:
        return
    
    # Vectorized processing for each feature in the block
    feat_offset = tl.arange(0, block_size)
    feat_mask = feat_offset + feat_start < n_features
    
    # Process each sequence position (vectorized across features)
    for seq_idx in range(n_seq_out):
        # Load data for all features in the block at once
        offset1 = (feat_offset + feat_start)[:, None] * n_seq1 + seq_idx
        offset2 = (feat_offset + feat_start)[:, None] * n_seq2 + seq_idx
        
        # Vectorized loads with broadcasting
        input1_vals = tl.load(input1_ptr + offset1, mask=feat_mask[:, None], other=0.0)
        input2_vals = tl.load(input2_ptr + offset2, mask=feat_mask[:, None], other=0.0)
        
        # Vectorized addition
        results = input1_vals + input2_vals
        
        # Store in transposed position (seq x features)
        output_offset = seq_idx * n_features + (feat_offset + feat_start)[:, None]
        tl.store(output_ptr + output_offset, results, mask=feat_mask[:, None])

@torch.fx.wrap
def fused_slice_add_transpose(tmp_5, tmp_4):
    # Input shapes
    _, n_features, n_seq1_orig = tmp_5.shape  # [1, 768, 125]
    _, _, n_seq2_orig = tmp_4.shape          # [1, 768, 235]
    n_seq_out = 124                           # We slice to 124 elements
    
    # Output shape after transpose: [1, 124, 768]
    output_shape = (1, n_seq_out, n_features)
    output = torch.empty(output_shape, dtype=tmp_5.dtype, device=tmp_5.device)
    
    # Use smaller blocks for better GPU occupancy and vectorization
    BLOCK_SIZE = 64  # Smaller blocks for better parallelization
    
    # Calculate number of blocks
    num_blocks = (n_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized vectorized kernel
    # Note: n_seq_out must be a compile-time constant, so we pass it directly
    vectorized_add_transpose_kernel[(num_blocks,)](
        tmp_5,
        tmp_4,
        output,
        n_features,
        124,  # n_seq_out is always 124, constant
        n_seq1_orig,
        n_seq2_orig,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_slice_add_transpose