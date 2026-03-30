import torch
import triton
import triton.language as tl

def pattern(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Match torch.nn.functional.embedding with the exact parameters used in the computation"""
    result = torch.nn.functional.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return result

def replacement_args(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Extract arguments needed for the optimized embedding lookup"""
    return input, weight

@triton.jit
def embedding_lookup_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    seq_len,
):
    """Simple and reliable embedding lookup kernel using Triton"""
    pid = tl.program_id(0)
    
    # Only process if within sequence length
    if pid < seq_len:
        # Load input index
        input_idx = tl.load(input_ptr + pid)
        
        if input_idx < num_embeddings:
            # Calculate base offset for this embedding
            base_offset = input_idx * embedding_dim
            
            # Load embedding vector element by element
            for j in range(embedding_dim):
                weight_value = tl.load(weight_ptr + base_offset + j)
                tl.store(output_ptr + pid * embedding_dim + j, weight_value)
        else:
            # Store zeros for invalid indices
            for j in range(embedding_dim):
                tl.store(output_ptr + pid * embedding_dim + j, 0.0)

@torch.fx.wrap
def optimized_embedding_lookup(input, weight):
    """Optimized embedding lookup using Triton kernel"""
    # Get input dimensions
    if input.dim() == 1:
        seq_len = input.shape[0]
        output_shape = (seq_len, weight.shape[1])
    else:
        batch_size, seq_len = input.shape
        output_shape = (batch_size, seq_len, weight.shape[1])
    
    num_embeddings = weight.shape[0]
    embedding_dim = weight.shape[1]
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Reshape input to be flat for processing (flatten batch dimension)
    if input.dim() == 2:
        flat_input = input.view(-1)  # shape: [batch_size * seq_len]
        flat_output = output.view(-1, embedding_dim)  # shape: [batch_size * seq_len, embedding_dim]
    else:
        flat_input = input
        flat_output = output
    
    total_elements = flat_input.shape[0]
    
    # Launch one program per sequence element
    embedding_lookup_kernel[(total_elements,)](
        input_ptr=flat_input,
        weight_ptr=weight,
        output_ptr=flat_output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        seq_len=total_elements,
    )
    
    return output

def replacement_func():
    """Return the optimized embedding lookup function"""
    return optimized_embedding_lookup