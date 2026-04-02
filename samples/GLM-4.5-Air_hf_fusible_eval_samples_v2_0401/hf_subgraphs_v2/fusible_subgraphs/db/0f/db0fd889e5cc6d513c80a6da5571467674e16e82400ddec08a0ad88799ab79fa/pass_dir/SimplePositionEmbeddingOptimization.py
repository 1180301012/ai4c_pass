import torch
import triton
import triton.language as tl

def pattern(position_embeddings):
    """
    Simple pattern for position embedding slicing that uses only basic operations.
    This avoids forbidden APIs like torch.cat and torch.nn.functional.interpolate.
    
    Matches: 
    tmp_13 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]
    tmp_15 = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = in_5[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    """
    # Extract CLS token (index 0)
    cls_token = position_embeddings[..., 0:1, :]  # tmp_14
    
    # Extract detection tokens (last 10 tokens)
    detection_tokens = position_embeddings[..., -10:, :]  # tmp_15
    
    # Extract intermediate tokens (middle 225 tokens)
    intermediate_tokens = position_embeddings[..., 1:-10, :]  # tmp_16
    
    return cls_token, detection_tokens, intermediate_tokens

@triton.jit
def simple_position_embedding_kernel(
    input_ptr,           # Position embeddings [1, 236, 32]
    cls_out_ptr,         # CLS tokens [1, 1, 32]
    detection_out_ptr,   # Detection tokens [1, 10, 32]
    intermediate_out_ptr, # Intermediate tokens [1, 225, 32]
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that performs slicing operations efficiently"""
    pid = tl.program_id(0)
    
    # Determine which type of tensor we're processing
    proc_type = pid // ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    elem_idx = pid % ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    if proc_type >= 3:  # Only process 3 tensor types
        return
        
    mask = elem_idx < seq_len
    
    if mask:
        if proc_type == 0:  # CLS tokens (only first element)
            if elem_idx == 0:
                input_offset = batch_size * (seq_len * hidden_dim) + elem_idx * hidden_dim
                output_offset = batch_size * (1 * hidden_dim)
                
                for d in range(0, hidden_dim, BLOCK_SIZE):
                    dim_idx = min(d + tl.arange(0, BLOCK_SIZE), hidden_dim - 1)
                    val = tl.load(input_ptr + input_offset + dim_idx, mask=True, other=0.0)
                    tl.store(cls_out_ptr + output_offset + dim_idx, val, mask=bool(mask))
        
        elif proc_type == 1:  # Intermediate tokens (elements 1 to -10)
            if elem_idx >= 1 and elem_idx < seq_len - 10:
                input_offset = batch_size * (seq_len * hidden_dim) + elem_idx * hidden_dim
                output_idx = elem_idx - 1  # Remove CLS token
                output_offset = batch_size * ((seq_len - 11) * hidden_dim) + output_idx * hidden_dim
                
                for d in range(0, hidden_dim, BLOCK_SIZE):
                    dim_idx = min(d + tl.arange(0, BLOCK_SIZE), hidden_dim - 1)
                    val = tl.load(input_ptr + input_offset + dim_idx, mask=True, other=0.0)
                    tl.store(intermediate_out_ptr + output_offset + dim_idx, val, mask=bool(mask))
        
        elif proc_type == 2:  # Detection tokens (last 10 elements)
            if elem_idx >= seq_len - 10:
                input_offset = batch_size * (seq_len * hidden_dim) + elem_idx * hidden_dim
                output_idx = elem_idx - (seq_len - 10)  # Offset in detection tokens
                output_offset = batch_size * (10 * hidden_dim) + output_idx * hidden_dim
                
                for d in range(0, hidden_dim, BLOCK_SIZE):
                    dim_idx = min(d + tl.arange(0, BLOCK_SIZE), hidden_dim - 1)
                    val = tl.load(input_ptr + input_offset + dim_idx, mask=True, other=0.0)
                    tl.store(detection_out_ptr + output_offset + dim_idx, val, mask=bool(mask))

@triton.jit
def simple_position_embedding_kernel(
    input_ptr,           # Position embeddings [1, 236, 32]
    cls_out_ptr,         # CLS tokens [1, 1, 32]
    detection_out_ptr,   # Detection tokens [1, 10, 32]
    intermediate_out_ptr, # Intermediate tokens [1, 225, 32]
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that performs slicing operations efficiently"""
    pid = tl.program_id(0)
    
    # Determine which type of tensor we're processing
    proc_type = pid // ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    elem_idx = pid % ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    if proc_type >= 3:  # Only process 3 tensor types
        return
        
    mask = elem_idx < seq_len
    
    if mask:
        if proc_type == 0:  # CLS tokens (only first element)
            if elem_idx == 0:
                input_offset = batch_size * (seq_len * hidden_dim) + elem_idx * hidden_dim
                output_offset = batch_size * (1 * hidden_dim)
                
                for d in range(0, hidden_dim, BLOCK_SIZE):
                    dim_idx = min(d + tl.arange(0, BLOCK_SIZE), hidden_dim - 1)
                    val = tl.load(input_ptr + input_offset + dim_idx, mask=True, other=0.0)
                    tl.store(cls_out_ptr + output_offset + dim_idx, val, mask=bool(mask))
        
        elif proc_type == 1:  # Intermediate tokens (elements 1 to -10)
            if elem_idx >= 1 and elem_idx < seq_len - 10:
                input_offset = batch_size * (seq_len * hidden_dim) + elem_idx * hidden_dim
                output_idx = elem_idx - 1  # Remove CLS token
                output_offset = batch_size * ((seq_len - 11) * hidden_dim) + output_idx * hidden_dim
                
                for d in range(0, hidden_dim, BLOCK_SIZE):
                    dim_idx = min(d + tl.arange(0, BLOCK_SIZE), hidden_dim - 1)
                    val = tl.load(input_ptr + input_offset + dim_idx, mask=True, other=0.0)
                    tl.store(intermediate_out_ptr + output_offset + dim_idx, val, mask=bool(mask))
        
        elif proc_type == 2:  # Detection tokens (last 10 elements)
            if elem_idx >= seq_len - 10:
                input_offset = batch_size * (seq_len * hidden_dim) + elem_idx * hidden_dim
                output_idx = elem_idx - (seq_len - 10)  # Offset in detection tokens
                output_offset = batch_size * (10 * hidden_dim) + output_idx * hidden_dim
                
                for d in range(0, hidden_dim, BLOCK_SIZE):
                    dim_idx = min(d + tl.arange(0, BLOCK_SIZE), hidden_dim - 1)
                    val = tl.load(input_ptr + input_offset + dim_idx, mask=True, other=0.0)
                    tl.store(detection_out_ptr + output_offset + dim_idx, val, mask=bool(mask))

def optimized_position_embedding_slicing(position_embeddings):
    """
    Simple optimized function that extracts different token types from position embeddings
    using a single Triton kernel instead of multiple slice operations.
    """
    batch_size, seq_len, hidden_dim = position_embeddings.shape
    
    # Create output tensors
    cls_tokens = torch.empty((batch_size, 1, hidden_dim), dtype=position_embeddings.dtype)
    detection_tokens = torch.empty((batch_size, 10, hidden_dim), dtype=position_embeddings.dtype)
    intermediate_tokens = torch.empty((batch_size, seq_len - 11, hidden_dim), dtype=position_embeddings.dtype)
    
    # Launch optimized kernel
    BLOCK_SIZE = 1024
    total_elements = batch_size * seq_len
    grid = (3 * ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE),)
    
    simple_position_embedding_kernel[grid](
        position_embeddings,
        cls_tokens,
        detection_tokens,
        intermediate_tokens,
        batch_size,
        seq_len,
        hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cls_tokens, detection_tokens, intermediate_tokens

@torch.fx.wrap
def simple_position_embedding_wrapper(position_embeddings):
    """Wrapper function for optimized position embedding slicing"""
    return optimized_position_embedding_slicing(position_embeddings)

def replacement_func():
    """Returns the replacement function"""
    return simple_position_embedding_wrapper