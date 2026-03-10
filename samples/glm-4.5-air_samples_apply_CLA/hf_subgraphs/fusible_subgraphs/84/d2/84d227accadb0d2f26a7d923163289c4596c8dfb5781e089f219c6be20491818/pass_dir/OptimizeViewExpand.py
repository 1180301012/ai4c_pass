import torch
import triton
import triton.language as tl

def slice_tensor(x):
    return x[slice(None, None, None), None, None, slice(None, None, None)]

def expand_tensor(x, *sizes):
    return x.expand(*sizes)

def pattern(x, size1, size2):
    # Simplified pattern to avoid proxy iteration issues
    tmp_9 = x[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_10 = tmp_9.expand(size1, 1, size2, x.shape[-1])
    return tmp_10

def replacement_args(x, size1, size2):
    return (x, size1, size2)

@triton.jit
def expand_kernel(
    output_ptr,
    input_ptr,
    batch_size,
    input_seq_len,
    output_batch_size,
    output_seq_len,
    hidden_size,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    # Each program handles output block
    out_batch_idx = tl.program_id(0)
    out_seq_idx = tl.program_id(1)
    
    # Calculate input indices (since we're expanding from [B, 1, S, H] to [B', S', H'])
    # The input is expanded along batch and sequence dimensions
    in_batch_idx = min(out_batch_idx, batch_size - 1)  # Clamp to valid input batch index
    in_seq_idx = min(out_seq_idx, input_seq_len - 1)    # Clamp to valid input sequence index
    
    # Calculate offsets
    input_offset = in_batch_idx * input_seq_len * hidden_size + in_seq_idx * hidden_size
    output_offset = out_batch_idx * output_seq_len * hidden_size + out_seq_idx * hidden_size
    
    # Load input data
    input_data = tl.load(input_ptr + input_offset)
    
    # Store expanded data
    tl.store(output_ptr + output_offset, input_data)

@torch.fx.wrap
def optimized_expand(x, *sizes):
    """
    Optimized expand operation that eliminates intermediate tensor creation
    for the specific pattern: [B, 1, S, H] -> [B_out, S_out, H]
    """
    original_shape = x.shape
    expanded_shape = sizes
    
    # Validate input shape matches expected pattern
    assert len(original_shape) == 4, f"Expected 4D input, got {len(original_shape)}D"
    assert original_shape[1] == 1, f"Expected size 1 in dim 1, got {original_shape[1]}"
    assert original_shape[3] == expanded_shape[-1], f"Hidden size mismatch: {original_shape[3]} vs {expanded_shape[-1]}"
    
    batch_size, _, input_seq_len, hidden_size = original_shape
    output_batch_size, output_seq_len = expanded_shape[0], expanded_shape[1]
    
    # Create output tensor
    output = torch.empty(expanded_shape, dtype=x.dtype, device=x.device)
    
    # Use simple indexing for small tensors or optimized kernel for larger ones
    if batch_size * input_seq_len < 1024:
        # For small tensors, use standard indexing with optimization
        output[out_batch_idx, out_seq_idx, :] = x[in_batch_idx, 0, in_seq_idx, :]
    else:
        # For larger tensors, use optimized kernel
        grid = (
            (output_batch_size + 63) // 64,
            (output_seq_len + 63) // 64,
        )
        
        expand_kernel[grid](
            output_ptr=output,
            input_ptr=x,
            batch_size=batch_size,
            input_seq_len=input_seq_len,
            output_batch_size=output_batch_size,
            output_seq_len=output_seq_len,
            hidden_size=hidden_size,
            BLOCK_SIZE_B=64,
            BLOCK_SIZE_S=64,
        )
    
    return output

def replacement_func():
    return optimized_expand

# Alternative pattern for when we have different original shapes
def pattern_flexible(x, *sizes):
    try:
        tmp_9 = x[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_10 = tmp_9.expand(*sizes)
        return tmp_10
    except:
        # Fallback for cases where the slicing pattern doesn't match
        raise NotImplementedError("This pattern only supports specific slicing/expansion")

def replacement_args_flexible(x, *sizes):
    return (x,) + sizes

@torch.fx.wrap  
def optimized_expand_flexible(x, *sizes):
    """
    More flexible version that handles different patterns
    """
    if x.dim() == 4 and x.shape[1] == 1:
        # Use optimized version for the main pattern
        return optimized_expand(x, *sizes)
    else:
        # For other patterns, create intermediate tensor but optimize the expansion
        tmp_9 = x[slice(None, None, None), None, None, slice(None, None, None)]
        # Use efficient expansion
        return tmp_9.expand(*sizes)

def replacement_func_flexible():
    return optimized_expand_flexible