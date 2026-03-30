import torch
import triton
import triton.language as tl

# Pattern that matches the slicing operations on tmp_5 and tmp_4
def pattern(tmp_5, tmp_4):
    # Both tensors are sliced to the same length (124)
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    return tmp_6, tmp_7

def replacement_args(tmp_5, tmp_4):
    return (tmp_5, tmp_4)

@triton.jit
def optimize_slicing_kernel(
    tmp5_ptr, tmp4_ptr, 
    out6_ptr, out7_ptr,
    n_features, n_seq,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one feature dimension
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for feature dimension
    feature_mask = offsets < n_features
    
    # Load from tmp_5 and tmp_4 for the current feature block
    # tmp_5 shape: [1, n_features, n_seq] -> load [n_features, 124]
    # tmp_4 shape: [1, n_features, n_seq] -> load [n_features, 124]
    seq_indices = tl.arange(0, 124)
    
    # Reshape offsets for proper tensor indexing
    feature_offsets = offsets[:, None]
    seq_offsets = seq_indices[None, :]
    
    # Combine offsets: [BLOCK_SIZE, 124]
    combined_offsets = feature_offsets + n_features * seq_offsets
    
    # Load data with proper broadcasting
    tmp5_data = tl.load(tmp5_ptr + combined_offsets, mask=feature_mask[:, None] & (seq_offsets < 124), other=0.0)
    tmp4_data = tl.load(tmp4_ptr + combined_offsets, mask=feature_mask[:, None] & (seq_offsets < 124), other=0.0)
    
    # Store results
    tl.store(out6_ptr + combined_offsets, tmp5_data, mask=feature_mask[:, None] & (seq_offsets < 124))
    tl.store(out7_ptr + combined_offsets, tmp4_data, mask=feature_mask[:, None] & (seq_offsets < 124))

@torch.fx.wrap
def optimized_slicing_operation(tmp_5, tmp_4):
    # Get tensor shapes
    batch_size, n_features, n_seq = tmp_5.shape
    
    # Output shapes
    out_shape = (batch_size, n_features, 124)
    
    # Create output tensors
    tmp_6 = torch.empty(out_shape, dtype=tmp_5.dtype, device=tmp_5.device)
    tmp_7 = torch.empty(out_shape, dtype=tmp_4.dtype, device=tmp_4.device)
    
    # Set up Triton kernel launch
    BLOCK_SIZE = 256
    num_features = (n_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_slicing_kernel[(num_features,)](
        tmp5_ptr=tmp_5,
        tmp4_ptr=tmp_4,
        out6_ptr=tmp_6,
        out7_ptr=tmp_7,
        n_features=n_features,
        n_seq=124,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return tmp_6, tmp_7

def replacement_func():
    return optimized_slicing_operation