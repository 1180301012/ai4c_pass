import torch
import triton
import triton.language as tl

@torch.fx.wrap
def fused_split_unsqueeze(input_tensor):
    """Main wrapper for the fused split + unsqueeze operation"""
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Define the split sizes based on the pattern
    split0_size, split1_size, split2_size = 512, 512, 128
    
    # Verify the tensor is large enough
    if hidden_dim < (split0_size + split1_size + split2_size):
        raise ValueError(f"Input tensor dimension {hidden_dim} is too small for split [{split0_size}, {split1_size}, {split2_size}]")
    
    # Calculate output shapes
    unsqueeze_shape = (batch_size, seq_len, split2_size, 1)
    chunk0_shape = (batch_size, seq_len, split0_size)
    chunk1_shape = (batch_size, seq_len, split1_size)
    
    # Create output tensors
    unsqueeze_out = torch.empty(unsqueeze_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    chunk0_out = torch.empty(chunk0_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    chunk1_out = torch.empty(chunk1_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions
    total_unsqueeze_elements = batch_size * seq_len * split2_size * 1
    total_chunk0_elements = batch_size * seq_len * split0_size
    total_chunk1_elements = batch_size * seq_len * split1_size
    
    BLOCK_SIZE = 128  # Optimal block size for GPU
    
    num_unsqueeze_programs = (total_unsqueeze_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_chunk0_programs = (total_chunk0_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_chunk1_programs = (total_chunk1_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for each output type
    # Use program_id(1) to distinguish between the three outputs
    grid = (
        max(total_unsqueeze_elements, total_chunk0_elements, total_chunk1_elements) // BLOCK_SIZE + 1,
        3,  # Three output types
    )
    
    fused_split_unsqueeze_kernel[grid](
        input_ptr=input_tensor,
        unsqueeze_out_ptr=unsqueeze_out,
        chunk0_out_ptr=chunk0_out,
        chunk1_out_ptr=chunk1_out,
        input_shape=(batch_size, seq_len, hidden_dim),
        slice_sizes=(split0_size, split1_size, split2_size),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return unsqueeze_out, chunk0_out, chunk1_out

@triton.jit
def fused_split_unsqueeze_kernel(
    input_ptr,
    unsqueeze_out_ptr,
    chunk0_out_ptr, 
    chunk1_out_ptr,
    input_shape,
    slice_sizes,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that splits and applies unsqueeze in one pass"""
    batch_size, seq_len, hidden_dim = input_shape
    
    # Calculate the split points
    split0_size = slice_sizes[0]
    split1_size = slice_sizes[1]
    split2_size = slice_sizes[2]
    
    # Each program processes a single element in output space
    # We need to handle three outputs: unsqueezed result, chunk0, chunk1
    out_idx = tl.program_id(0)
    
    # Determine which output we're processing: 0=unsqueeze, 1=chunk0, 2=chunk1
    output_type = tl.program_id(1)
    
    if output_type == 0:  # Process unsqueezed result
        # Calculate input offset for third chunk (hidden_dim - split2_size to end)
        input_offset0 = out_idx // (split2_size * 1) * (batch_size * seq_len * hidden_dim)
        input_offset1 = (out_idx % (split2_size * 1) // 1) * (seq_len * hidden_dim)
        input_offset2 = (out_idx % 1) * hidden_dim + (hidden_dim - split2_size)
        
        input_idx = input_offset0 + input_offset1 + input_offset2
        
        # Load from input (third chunk)
        input_val = tl.load(input_ptr + input_idx, other=0.0)
        
        # Store to unsqueeze output (this has an added dimension)
        unsqueeze_row = out_idx // split2_size
        unsqueeze_col = out_idx % split2_size
        unsqueeze_idx = unsqueeze_row * (batch_size * seq_len * split2_size * 1) + \
                       unsqueeze_col
        tl.store(unsqueeze_out_ptr + unsqueeze_idx, input_val, other=0.0)
        
    elif output_type == 1:  # Process first chunk [0:split0_size]
        split0_start = 0
        input_row = out_idx // (split0_size * seq_len) 
        input_col = (out_idx % (split0_size * seq_len)) // split0_size
        input_depth = (out_idx % split0_size)
        
        input_idx = input_row * (seq_len * hidden_dim) + \
                   input_col * hidden_dim + \
                   input_depth
        
        output_idx = input_row * (seq_len * split0_size) + \
                    input_col * split0_size + \
                    input_depth
        
        input_val = tl.load(input_ptr + input_idx, other=0.0)
        tl.store(chunk0_out_ptr + output_idx, input_val, other=0.0)
        
    else:  # Process second chunk [split0_size:split0_size+split1_size]
        split0_size_int = slice_sizes[0]
        split1_start = split0_size_int
        
        input_row = out_idx // (split1_size * seq_len)
        input_col = (out_idx % (split1_size * seq_len)) // split1_size
        input_depth = out_idx % split1_size + split1_start
        
        input_idx = input_row * (seq_len * hidden_dim) + \
                   input_col * hidden_dim + \
                   input_depth
        
        output_idx = input_row * (seq_len * split1_size) + \
                    input_col * split1_size + \
                    (input_depth - split1_start)
        
        input_val = tl.load(input_ptr + input_idx, other=0.0)
        tl.store(chunk1_out_ptr + output_idx, input_val, other=0.0)

def pattern(input_tensor):
    """Pattern: Split tensor into chunks and apply unsqueeze to the third chunk"""
    # First split the input tensor along dimension 2 into [512, 512, 128] chunks
    split_result = torch.split(input_tensor, [512, 512, 128], dim=2)
    # Extract the third chunk (index 2) and apply unsqueeze(2)
    result = split_result[2].unsqueeze(2)
    # Return both the unsqueezed result and the other two chunks for proper dataflow
    return result, split_result[0], split_result[1]

def replacement_args(input_tensor):
    """Extract arguments needed for replacement - just the input tensor"""
    return (input_tensor,)

def replacement_func():
    """Return the fused kernel implementation"""
    return fused_split_unsqueeze