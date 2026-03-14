import torch
import triton
import triton.language as tl

def pattern(in_2):
    # Match the slice + transpose + reshape pattern
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 128, 96, 96)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def slice_transpose_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    target_channels,
    target_height,
    target_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID mapping
    pid = tl.program_id(axis=0)
    
    # Calculate output block offsets
    m_offset = pid * BLOCK_SIZE_M
    m_coords = m_offset + tl.arange(0, BLOCK_SIZE_M)
    
    # Mask for output bounds checking
    m_mask = m_coords < target_height
    
    # Calculate input coordinates from output coordinates
    # We need to unfold the target dimensions back to input dimensions
    # Output shape: [1, 128, 96, 96]
    # Maps from: [batch=0, channel=0-127, h=0-95, w=0-95]
    # To input slice: [batch=0, head=8, seq_idx=1-9216, head_dim=16] -> transpose -> [batch=0, head=8, head_dim=16, seq_idx=1-9216] -> reshape
    
    # For each output element, calculate corresponding input element
    for i, m in enumerate(m_coords):
        if m < target_height:
            # Calculate channel and position in target shape
            remaining = m
            h = remaining // target_width
            remaining = remaining % target_width
            w = remaining
            
            # Map back to reshaped dimensions: [1, 128, 96, 96] 
            # 128 = num_heads * head_dim = 8 * 16
            # So: channel_coord = h_coord * target_width + w_coord
            output_idx = h * target_width + w
            
            # Map to transpose dimensions: [1, 8, 16, 9216]
            head_idx = output_idx // head_dim  # 0-7
            head_dim_idx = output_idx % head_dim  # 0-15
            seq_idx = h  # mapped from target_height (96) -> seq dimension
            
            # Map to original slice: [1, 8, 9216, 16]
            original_seq_idx = seq_idx + 1  # +1 because we sliced from index 1
            
            # Only process if within bounds
            if head_idx < num_heads and seq_idx < seq_len and original_seq_idx < seq_len + 1:
                # Calculate input offset
                input_offset = (batch_size * num_heads * seq_len * head_dim + 
                              head_idx * seq_len * head_dim + 
                              original_seq_idx * head_dim + 
                              head_dim_idx)
                
                # Calculate output offset  
                output_offset = (m * target_width)
                
                # Load from input and store to output
                val = tl.load(input_ptr + input_offset, other=0.0)
                tl.store(output_ptr + output_offset, val, mask=m_mask[i])

@torch.fx.wrap
def optimized_slice_transpose_reshape(in_2):
    # Get input shape
    batch_size, num_heads, seq_len_total, head_dim = in_2.shape
    
    # Calculate slice size (start from index 1)
    seq_len = seq_len_total - 1
    
    # Target output shape
    target_channels = 128  # 8 heads * 16 head_dim
    target_height = 96
    target_width = 96
    
    # Create output tensor
    output_shape = (1, target_channels, target_height, target_width)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Block sizes
    BLOCK_SIZE_M = 32
    
    # Calculate grid size
    grid_size = (target_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    slice_transpose_reshape_kernel[grid_size](
        in_2, output,
        batch_size, num_heads, seq_len_total, head_dim,
        target_channels, target_height, target_width,
        BLOCK_SIZE_M
    )
    
    return output

def replacement_func():
    return optimized_slice_transpose_reshape