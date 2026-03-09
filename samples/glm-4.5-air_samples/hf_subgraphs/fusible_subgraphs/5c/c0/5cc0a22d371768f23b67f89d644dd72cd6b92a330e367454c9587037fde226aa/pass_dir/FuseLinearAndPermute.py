import torch
import triton
import triton.language as tl

# Pattern matching function - matches permute operation
def pattern(x):
    return x.permute(0, 3, 1, 2)


# Argument extraction function
def replacement_args(x):
    return (x,)


# Optimized kernel for permute operation: [batch, seq1, seq2, features] -> [batch, features, seq1, seq2]
@triton.jit
def optimized_permute_kernel(
    x_ptr,
    out_ptr,
    n_batch, n_seq1, n_seq2, n_features,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Kernel for permute operation that correctly rearranges dimensions:
    [batch, seq1, seq2, features] -> [batch, features, seq1, seq2]
    """
    # Get program ID for batch processing
    pid = tl.program_id(0)
    
    # Create offsets for elements in this program
    offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < (n_batch * n_seq1 * n_seq2 * n_features)
    
    # Convert linear offset to multi-dimensional coordinates
    # Input: [batch, seq1, seq2, features]
    batch_size = n_seq1 * n_seq2 * n_features
    seq1_size = n_seq2 * n_features
    seq2_size = n_features
    
    batch_idx = offsets // batch_size
    remainder = offsets % batch_size
    seq1_idx = remainder // seq1_size
    remainder = remainder % seq1_size
    seq2_idx = remainder // seq2_size
    feature_idx = remainder % seq2_size
    
    # Validate indices
    batch_mask = batch_idx < n_batch
    seq1_mask = seq1_idx < n_seq1
    seq2_mask = seq2_idx < n_seq2
    feature_mask = feature_idx < n_features
    
    valid_mask = batch_mask & seq1_mask & seq2_mask & feature_mask
    
    # Calculate output coordinates
    # Output: [batch, features, seq1, seq2]
    output_stride_batch = n_features * n_seq1 * n_seq2
    output_stride_features = n_seq1 * n_seq2
    output_stride_seq1 = n_seq2
    
    output_offset = (batch_idx * output_stride_batch + 
                    feature_idx * output_stride_features + 
                    seq1_idx * output_stride_seq1 + 
                    seq2_idx)
    
    # Load from input and store to output with correct permutation
    input_val = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
    tl.store(out_ptr + output_offset, input_val, mask=valid_mask)

@torch.fx.wrap
def optimized_permute(x):
    """Optimized permute operation: [batch, seq1, seq2, features] -> [batch, features, seq1, seq2]"""
    batch, seq1, seq2, features = x.shape
    
    # Choose optimized block size for better performance (must be power of 2)
    total_elements = batch * seq1 * seq2 * features
    optimal_block_size = min(1024, max(256, total_elements // 512))  # Adaptive block sizing
    # Round to nearest power of 2
    BLOCK_SIZE_M = 1 << (optimal_block_size.bit_length() - 1) if optimal_block_size > 1 else 1
    
    # Calculate grid dimensions
    total_elements = batch * seq1 * seq2 * features
    num_programs = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Create output tensor in permuted shape [batch, features, seq1, seq2]
    output = torch.empty((batch, features, seq1, seq2), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    optimized_permute_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=output,
        n_batch=batch,
        n_seq1=seq1,
        n_seq2=seq2,
        n_features=features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return output

def replacement_func():
    return optimized_permute