import torch
import triton
import triton.language as tl

def pattern(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, **kwargs):
    """
    Pattern to match: embedding operation followed by slice
    """
    # Embedding operation
    embedded = torch.nn.functional.embedding(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    # Slice operation - we need to handle different slice patterns
    embedded_slice = embedded[slice(None, None, None), slice(1, None, None)]
    return embedded, embedded_slice

def replacement_args(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, **kwargs):
    """
    Extract arguments needed for the fused operation
    """
    input_ids = input_ids.contiguous()
    weight = weight.contiguous()
    
    # Get slice information - this is a specific slice pattern we're optimizing
    # slice(None, None, None), slice(1, None, None) - removes first element along dim 1
    start_dim1 = 1
    end_dim1 = None
    step_dim1 = None
    
    return (input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, 
            start_dim1, end_dim1, step_dim1)

@triton.jit
def fused_embedding_slice_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    partial_output_ptr,
    input_ids_size,
    embedding_dim,
    vocab_size,
    start_dim1,
    end_dim1,
    step_dim1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused embedding + slice kernel that directly computes only the needed part
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_ids_size
    
    # Load input IDs
    input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    
    # Compute embedding slice directly - only for indices >= start_dim1
    # For each input ID, we'll only compute the embedding vectors from start_dim1 onwards
    for i in range(BLOCK_SIZE):
        if offsets[i] < input_ids_size and input_ids[i] >= 0:  # Skip padding indices
            # Get the embedding vector
            emb_ptr = weight_ptr + input_ids[i] * embedding_dim
            # For the output, we need the full embedding
            full_emb = tl.load(emb_ptr + tl.arange(0, embedding_dim), other=0.0)
            
            # Store full embedding in the first output
            tl.store(output_ptr + offsets[i] * embedding_dim + tl.arange(0, embedding_dim), full_emb, mask=tl.arange(0, embedding_dim) < embedding_dim)
            
            # For the slice output, only store from start_dim1 onwards
            if start_dim1 < embedding_dim:
                slice_length = embedding_dim - start_dim1
                tl.store(partial_output_ptr + offsets[i] * slice_length + tl.arange(0, slice_length), 
                        full_emb[start_dim1:start_dim1 + slice_length], 
                        mask=tl.arange(0, slice_length) < slice_length)

@torch.fx.wrap
def fused_embedding_slice_function(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, start_dim1, end_dim1, step_dim1):
    """
    Fused function that computes embedding + slice in one kernel
    """
    input_ids_size = input_ids.numel()
    embedding_dim = weight.size(1)
    vocab_size = weight.size(0)
    
    # Create output tensors
    full_output = torch.empty(input_ids_size, embedding_dim, dtype=weight.dtype, device=weight.device)
    slice_output_size = embedding_dim - start_dim1 if start_dim1 < embedding_dim else 0
    slice_output = torch.empty(input_ids_size, slice_output_size, dtype=weight.dtype, device=weight.device) if slice_output_size > 0 else torch.empty(0, dtype=weight.dtype, device=weight.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    num_programs = (input_ids_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_slice_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=full_output,
        partial_output_ptr=slice_output,
        input_ids_size=input_ids_size,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        start_dim1=start_dim1,
        end_dim1=end_dim1,
        step_dim1=step_dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return full_output, slice_output

def replacement_func():
    """
    Return the fused function
    """
    return fused_embedding_slice_function