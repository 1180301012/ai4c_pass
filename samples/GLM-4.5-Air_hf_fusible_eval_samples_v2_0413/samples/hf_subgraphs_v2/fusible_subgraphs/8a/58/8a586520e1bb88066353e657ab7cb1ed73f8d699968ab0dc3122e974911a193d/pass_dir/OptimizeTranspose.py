import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Pattern matching for transpose operation
    Matches: in_2.transpose(-2, -1)
    """
    tmp_4 = in_2.transpose(-2, -1)
    return tmp_4

def replacement_args(in_2):
    """Extract arguments for the transpose kernel"""
    return (in_2,)

@triton.jit
def optimized_transpose_kernel(
    x_ptr,         # Input: [B, C, H, W] = [1, 16, 196, 48]
    out_ptr,       # Output: [B, C, W, H] = [1, 16, 48, 196]
    B,             # Batch size = 1
    C,             # Channels = 16
    H,             # Height = 196
    W,             # Width = 48
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for transpose operation H <-> W"""
    pid = tl.program_id(0)
    
    # Calculate offsets for parallel processing
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < B * C * H * W  # We process B*C*H*W elements in parallel
    
    # Calculate which element this thread is processing
    linear_idx = offset
    
    # Convert linear index to [B, C, H, W] coordinates
    batch_idx = linear_idx // (C * H * W)
    remainder = linear_idx % (C * H * W)
    channel_idx = remainder // (H * W)
    remainder = remainder % (H * W)
    pos_idx = remainder // W  # This will be the height position
    col_idx = remainder % W  # This will be the width position
    
    # For transpose, we swap H and W dimensions
    # Original stride: [C*H*W, H*W, W, 1]
    # Transposed stride: [C*W*H, W*H, H, 1] = [C*H*W, H*W, H, 1]
    transposed_linear_idx = batch_idx * (C * H * W) + channel_idx * (H * W) + col_idx * H + pos_idx
    
    # Bounds checking for the element
    element_mask = (pos_idx < H) & (col_idx < W)
    
    # Load from input and store to output (only for valid elements)
    if element_mask:
        input_val = tl.load(x_ptr + linear_idx, other=0.0)
        tl.store(out_ptr + transposed_linear_idx, input_val)

@torch.fx.wrap  
def optimized_transpose(in_2):
    """Wrapper function for optimized transpose operation"""
    B, C, H, W = in_2.shape
    
    # Use a block size that efficiently utilizes GPU threads
    # Based on dimensions, we want to process as many elements as possible in parallel
    BLOCK_SIZE = 1024  # Standard block size for good GPU utilization
    
    # Calculate number of programs needed
    total_elements = B * C * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((B, C, W, H), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    optimized_transpose_kernel[num_programs](
        in_2, out, B, C, H, W, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    """Return the optimized transpose function"""
    return optimized_transpose