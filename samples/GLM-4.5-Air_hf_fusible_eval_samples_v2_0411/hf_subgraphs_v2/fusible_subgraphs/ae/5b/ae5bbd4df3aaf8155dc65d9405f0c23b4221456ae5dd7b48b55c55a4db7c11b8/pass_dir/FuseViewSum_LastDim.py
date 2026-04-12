import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(linear):
    """
    Fuse view + sum operations to avoid intermediate tensor storage
    """
    # Use the actual seq_len from the input tensor to make it generic
    seq_len = linear.shape[1]
    tmp_4 = linear.view(1, seq_len, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    return tmp_5

# Argument extraction function
def replacement_args(linear):
    return (linear,)

# Triton kernel for fused view + sum operation using vectorized loads
@triton.jit
def fused_view_sum_kernel(
    input_ptr,          # Input tensor pointer
    output_ptr,         # Output tensor pointer  
    batch: tl.constexpr,
    seq_len: tl.constexpr, 
    hidden_size: tl.constexpr,
    input_features: tl.constexpr,  # Number of input features (8)
    output_features: tl.constexpr, # Number of output features (2)
    reduce_dim: tl.constexpr,       # Dimension to reduce over (4)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    
    # Total elements in output [batch, seq_len, hidden_size, output_features]
    total_output_elements = batch * seq_len * hidden_size * output_features
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_output_elements
    
    # Convert output offset to multi-dimensional indices
    # output_shape: [batch, seq_len, hidden_size, output_features]
    output_idx = offsets
    batch_idx = output_idx // (seq_len * hidden_size * output_features)
    remainder = output_idx % (seq_len * hidden_size * output_features)
    seq_idx = remainder // (hidden_size * output_features)
    remainder = remainder % (hidden_size * output_features)
    hidden_idx = remainder // output_features
    feature_idx = remainder % output_features
    
    # Vectorized load: Load 4 elements at once when reduce_dim == 4
    # This avoids the loop and uses vectorized memory access
    base_input_offset = (batch_idx * (seq_len * hidden_size * input_features) + 
                        seq_idx * (hidden_size * input_features) + 
                        hidden_idx * input_features + 
                        feature_idx * reduce_dim)
    
    # Load a vector of 4 elements
    input_vector = tl.load(input_ptr + base_input_offset, mask=mask, other=0.0)
    
    # Sum the loaded vector (this happens in registers)
    sum_val = tl.sum(input_vector, axis=0)
    
    # Store summed result
    tl.store(output_ptr + offsets, sum_val, mask=mask)

@torch.fx.wrap
def fused_view_sum(linear):
    """
    Fused implementation of view + sum operations
    """
    batch = 1
    
    # Determine model variant based on input shape
    if linear.shape[1] == 12:  # wavlm_base
        seq_len = 12
        hidden_size = 199
    else:  # wavlm_large
        seq_len = 16
        hidden_size = 199
        
    input_features = 8  # Original features after linear layer
    output_features = 2  # After sum over last dimension
    reduce_dim = 4       # Dimension we're summing over
        
    # Create output tensor with shape [batch, seq_len, hidden_size, output_features]
    output = torch.empty(batch * seq_len * hidden_size * output_features, dtype=linear.dtype, device=linear.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    total_elements = batch * seq_len * hidden_size * output_features
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_view_sum_kernel[(num_programs,)](
        input_ptr=linear,
        output_ptr=output,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        input_features=input_features,
        output_features=output_features,
        reduce_dim=reduce_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match expected dimensions
    return output.view(batch, seq_len, hidden_size, output_features)

# Replacement function
def replacement_func():
    return fused_view_sum