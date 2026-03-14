import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Match the computation pattern: add + permute(0,2,1) + view reshape"""
    # Match the exact sequence from the model
    tmp_0 = y + x
    tmp_1 = tmp_0.permute(0, 2, 1)
    # Use view with -1 for dimension inference during symbolic tracing
    tmp_2 = tmp_1.view(1, -1, -1, -1)
    return (tmp_2,)

def replacement_args(x, y):
    """Extract arguments for the replacement kernel"""
    return (x, y)

@triton.jit
def fused_add_permute_view_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    original_channels: tl.constexpr,
    original_seq_len: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Element-wise addition
    2. Permute(0, 2, 1) - swap dimensions 1 and 2  
    3. Reshape to 4D tensor
    """
    # Each program handles one row of the output tensor
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Ensure we don't go out of bounds
    if row_idx >= batch_size * spatial_size or col_idx >= spatial_size:
        return
    
    # Calculate global position in output
    output_offset = row_idx * spatial_size + col_idx
    
    # Calculate corresponding input positions considering the permute
    # Original dimensions: [batch_size, original_seq_len, original_channels]
    # Permute to: [batch_size, original_channels, original_seq_len] 
    # Then view to: [batch_size, original_channels, spatial_size, spatial_size]
    
    # Extract batch index
    batch_idx = row_idx // (original_channels * spatial_size)
    
    # Extract channel index within the batch's portion
    flat_idx = row_idx % (original_channels * spatial_size)
    channel_idx = flat_idx // spatial_size
    
    # Calculate the 2D spatial position in the original sequence
    spatial_row = col_idx
    spatial_col = flat_idx % spatial_size
    
    # Calculate linear index in the original (permuted) sequence dimension
    original_seq_idx = spatial_row * spatial_size + spatial_col
    
    # Final indices for input tensors (after permutation transformation)
    x_offset = batch_idx * original_seq_len * original_channels + original_seq_idx * original_channels + channel_idx
    y_offset = batch_idx * original_seq_len * original_channels + original_seq_idx * original_channels + channel_idx
    out_offset = batch_idx * original_channels * spatial_size * spatial_size + channel_idx * spatial_size * spatial_size + spatial_row * spatial_size + spatial_col
    
    # Load, add, and store
    x_val = tl.load(x_ptr + x_offset, mask=(batch_idx < batch_size), other=0.0)
    y_val = tl.load(y_ptr + y_offset, mask=(batch_idx < batch_size), other=0.0)
    out_val = x_val + y_val
    
    tl.store(out_ptr + out_offset, out_val, mask=(batch_idx < batch_size))

@torch.fx.wrap
def fused_add_permute_view(x, y):
    """
    Perform fused add + permute(0,2,1) + view operation in a single kernel
    """
    # Get input shapes
    batch_size, seq_len, channels = x.shape
    
    # Calculate spatial dimensions (assuming square spatial layout)
    # Handle both symbolic tracing and concrete execution
    try:
        spatial_size = int(seq_len ** 0.5)
        if spatial_size * spatial_size != seq_len:
            raise ValueError(f"Sequence length {seq_len} is not a perfect square")
    except:
        # During symbolic tracing we need to defer this calculation
        # The actual spatial size will be determined from the context
        seq_len_val = seq_len.item() if hasattr(seq_len, 'item') else int(seq_len)
        spatial_size = int(seq_len_val ** 0.5)
        if spatial_size * spatial_size != seq_len_val:
            raise ValueError(f"Sequence length {seq_len_val} is not a perfect square")
    
    # Create output tensor
    output_shape = (batch_size, channels, spatial_size, spatial_size)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid configuration
    total_rows = batch_size * channels * spatial_size
    total_cols = spatial_size
    
    BLOCK_SIZE = 256  # Optimized block size for memory coalescing
    num_rows = (total_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_cols = (total_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with proper 3D grid
    fused_add_permute_view_kernel[(num_rows, num_cols, 1)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        original_channels=channels,
        original_seq_len=seq_len,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused function reference"""
    return fused_add_permute_view