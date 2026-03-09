import torch
import triton
import triton.language as tl

@triton.jit
def optimized_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    items_per_program: tl.constexpr,
):
    """
    Optimized kernel for unsqueeze operations.
    Adds a dimension at position 1 efficiently.
    """
    pid = tl.program_id(0)
    
    # Calculate global start index for this program
    global_start_idx = pid * items_per_program
    
    # Generate offsets within this program's range
    offsets = tl.arange(0, items_per_program)
    
    # Calculate global indices
    indices = global_start_idx + offsets
    
    # Calculate mask for valid indices
    total_elements = batch_size * seq_len * embed_dim
    mask = indices < total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + indices, mask=mask, other=0.0)
    
    # Store output data (unsqueeze is handled at wrapper level)
    tl.store(output_ptr + indices, input_data, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze(input):
    """
    Optimized function for unsqueeze operations.
    Efficiently adds a dimension at position 1.
    """
    # For the embeddings we have, they are already 3D tensors
    # Shape: [batch_size, seq_len, embed_dim] -> [batch_size, 1, seq_len, embed_dim]
    input_shape = input.shape
    batch_size, seq_len, embed_dim = input_shape
    output_shape = (batch_size, 1, seq_len, embed_dim)
    
    # Create output with new dimension
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Use direct assignment for unsqueeze operation (simpler and more efficient)
    output[:, 0, :, :] = input
    
    return output

def pattern(cos_or_sin_embedding):
    """
    Pattern: Unsqueeze operation for cos/sin embeddings.
    """
    return cos_or_sin_embedding.unsqueeze(1)

def replacement_args(cos_or_sin_embedding):
    """Extract arguments for the replacement function"""
    return (cos_or_sin_embedding,)

def replacement_func():
    """Return the optimized unsqueeze function"""
    return optimized_unsqueeze