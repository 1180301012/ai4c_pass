import torch
import triton
import triton.language as tl

# Pattern matching function for slice + transpose + reshape sequence
def pattern(in_0, in_1, in_2):
    # Directly compose the operations without intermediate variables
    return in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)].transpose(-1, -2).reshape(1, 128, 96, 96)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_2,)  # We only need in_2 for the fused operation

# Optimized Triton kernel for fused slice + transpose + reshape
@triton.jit
def slice_transpose_reshape_kernel(
    input_ptr,       # in_2 tensor
    output_ptr,      # reshaped output tensor [1, C, H, W]
    n_batch,         # batch size (always 1)
    n_head,          # number of heads
    n_seq,           # sequence length after slicing (n_seq_orig - 1)
    n_embed,         # embedding dimension
    c_out,           # output channels
    h_out,           # output height
    w_out,           # output width
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs for parallel execution
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_channel = tl.program_id(2)
    pid_spatial = tl.program_id(3)
    
    # Map spatial coordinate to original input coordinates
    # We need to read from [1, n_head, n_embed, n_seq] and write to [1, c_out, h_out, w_out]
    # The total elements per head is n_embed * n_seq
    elements_per_head = n_embed * n_seq
    
    # Calculate which element in the head we're processing
    if pid_channel >= c_out:
        return
    
    spatial_offset = pid_spatial
    channel_offset = pid_channel * (h_out * w_out)
    total_offset = channel_offset + spatial_offset
    
    # Map to source coordinates (after transpose: [embed, seq] mapping)
    seq_idx = total_offset // n_embed
    embed_idx = total_offset % n_embed
    
    # Bounds checking
    if seq_idx >= n_seq or embed_idx >= n_embed:
        return
    
    # Calculate input pointer offset (transposed layout: [batch, head, embed, seq])
    input_offset = (pid_head * elements_per_head + embed_idx * n_seq + seq_idx)
    
    # Calculate output pointer offset (reshape layout: [batch, channel, height, width])
    output_offset = (pid_channel * h_out * w_out + spatial_offset)
    
    # Load and store data
    value = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, value)

# Kernel wrapper function
@torch.fx.wrap
def slice_transpose_reshape_fused(in_2):
    """
    Fused kernel that performs slicing, transpose, and reshape in one operation
    Specifically for target shape [1, 128, 96, 96]
    
    Args:
        in_2: Input tensor of shape [1, n_head, n_seq, n_embed] (before slicing)
    """
    # Target shape for this specific pass
    c_out, h_out, w_out = 128, 96, 96
    
    # Get input tensor properties (before slicing)
    n_batch, n_head, n_seq_orig, n_embed = in_2.shape
    
    # After slicing, sequence length becomes n_seq = n_seq_orig - 1
    n_seq = n_seq_orig - 1
    
    # Output tensor
    output = torch.empty((1, c_out, h_out, w_out), dtype=in_2.dtype, device=in_2.device)
    
    # Setup grid dimensions for parallel execution
    # Each program handles one element in the output tensor
    batch_size = n_batch  # Always 1
    head_dim = n_head      # Process each head in parallel
    channel_dim = c_out    # Process each output channel in parallel  
    spatial_dim = h_out * w_out  # Process spatial positions in parallel
    
    # Create grid
    grid = (batch_size, head_dim, channel_dim, spatial_dim)
    
    # Block size for fine-grained parallelism
    BLOCK_SIZE = 32
    
    # Launch kernel
    slice_transpose_reshape_kernel[grid](
        input_ptr=in_2.data_ptr(),
        output_ptr=output.data_ptr(),
        n_batch=n_batch,
        n_head=n_head,
        n_seq=n_seq,
        n_embed=n_embed,
        c_out=c_out,
        h_out=h_out,
        w_out=w_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns the optimized function)
def replacement_func():
    return slice_transpose_reshape_fused