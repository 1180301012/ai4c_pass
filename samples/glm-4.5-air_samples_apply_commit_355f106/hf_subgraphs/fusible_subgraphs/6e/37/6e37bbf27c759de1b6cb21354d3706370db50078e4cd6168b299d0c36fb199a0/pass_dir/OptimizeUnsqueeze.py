import torch
import triton
import triton.language as tl

@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    feature_dim: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    
    # Calculate output and input indices
    # Output shape: [batch, 1, seq_len, feature_dim]
    # Input shape: [batch, seq_len, feature_dim]
    
    # Batch dimension
    batch_idx = pid // (seq_len * feature_dim)
    # Sequence dimension (now at position 2 in output)
    seq_idx = (pid // feature_dim) % seq_len
    # Feature dimension
    feat_idx = pid % feature_dim
    
    # Check bounds
    mask = (batch_idx < batch) & (seq_idx < seq_len) & (feat_idx < feature_dim)
    
    if not mask:
        return
    
    # Load from input at [batch_idx, seq_idx, feat_idx]
    input_offset = batch_idx * seq_len * feature_dim + seq_idx * feature_dim + feat_idx
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output at [batch_idx, 0, seq_idx, feat_idx]
    # The second dimension (unsqueeze) is always 0, so we need to adjust the indexing
    # Total elements in output: batch * 1 * seq_len * feature_dim
    output_offset = batch_idx * seq_len * feature_dim + seq_idx * feature_dim + feat_idx
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze(x):
    """
    Optimized unsqueeze operation that adds dimension at position 1
    Input: [batch, seq_len, feature_dim]
    Output: [batch, 1, seq_len, feature_dim]
    """
    *shape, last_dim = x.shape
    
    # For the specific case in our graphs, the input is [1, seq_len, feature_dim]
    # and we want [1, 1, seq_len, feature_dim]
    if len(shape) == 3:
        batch, seq_len, feature_dim = shape
        output_shape = (batch, 1, seq_len, feature_dim)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        
        # Set up grid dimensions
        total_elements = batch * seq_len * feature_dim
        grid_size = (total_elements + 1023) // 1024  # Using 1024 as block size
        
        # Launch kernel
        unsqueeze_kernel[grid_size](
            x,
            output,
            batch, seq_len, feature_dim
        )
        
        return output
    else:
        # Fallback for non-3D tensors
        return x.unsqueeze(1)

def pattern(x):
    """
    Pattern: unsqueeze(1) operation
    """
    # Simple unsqueeze operation
    return x.unsqueeze(1)

def replacement_args(x):
    """
    Extract arguments needed for the replacement function
    """
    return (x,)

def replacement_func():
    """
    Return the optimized kernel function
    """
    return optimized_unsqueeze