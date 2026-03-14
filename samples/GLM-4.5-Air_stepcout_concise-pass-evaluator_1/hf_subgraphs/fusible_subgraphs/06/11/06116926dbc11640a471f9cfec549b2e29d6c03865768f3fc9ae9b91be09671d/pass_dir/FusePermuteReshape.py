import torch
import triton
import triton.language as tl

def pattern(in_tensor, batch_size, C_out, H, W):
    """
    Pattern matching: permute + reshape operations
    Original: tmp_2 = in_tensor.permute(0, 2, 1)
              tmp_3 = tmp_2.reshape(batch_size, C_out, H, W)
    We fuse these into a single operation that directly produces the result
    """
    # Check if the reshape is valid
    expected_size = C_out * H * W
    actual_size = in_tensor.shape[1]  # This should be seq_len
    
    if expected_size != actual_size:
        raise ValueError(f"Size mismatch: expected {expected_size}, got {actual_size}")
        
    tmp_2 = in_tensor.permute(0, 2, 1)  # [batch_size, C_out, seq_len]
    tmp_3 = tmp_2.reshape(batch_size, C_out, H, W)
    return tmp_3

def replacement_args(in_tensor, batch_size, C_out, H, W):
    """
    Extract arguments needed for the replacement
    """
    return (in_tensor, batch_size, C_out, H, W)

@triton.jit
def optimized_permute_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    C_out,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that directly performs permute + reshape operations
    This avoids intermediate storage and memory allocation
    """
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len * C_out)
    
    # Process only valid offsets
    for i in range(BLOCK_SIZE):
        offset = offsets[i]
        if offset < (batch_size * seq_len * C_out):
            # Convert linear offset to 3D coordinates [batch, seq, C_out]
            batch_idx = offset // (seq_len * C_out)
            seq_idx = (offset % (seq_len * C_out)) // C_out
            feat_idx = offset % C_out
            
            # Permute + reshape: [batch, seq, C_out] -> [batch, C_out, H, W]
            # This means: output = [batch, C_out, H, W] where seq = H * W
            output_offset = (batch_idx * C_out * H * W + 
                           feat_idx * H * W + 
                           seq_idx)
            
            # Load from input and store to output
            val = tl.load(input_ptr + offset, mask=True)
            tl.store(output_ptr + output_offset, val)

@torch.fx.wrap
def optimized_permute_reshape(in_tensor, batch_size, C_out, H, W):
    """
    Optimized function that fuses permute + reshape operations
    """
    # Get input dimensions
    seq_len = in_tensor.shape[1]
    
    # Create output tensor with the desired shape: [batch_size, C_out, H, W]
    output_shape = (batch_size, C_out, H, W)
    out = torch.empty(output_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    
    # Set up grid dimensions - use 1D grid for simplicity
    total_elements = batch_size * seq_len * C_out
    BLOCK_SIZE = 256  # Smaller block size for better branch divergence handling
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_permute_reshape_kernel[(num_programs,)](
        in_tensor,
        out,
        batch_size,
        seq_len,
        C_out,
        H,
        W,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    """
    Returns the optimized function that replaces the pattern
    """
    return optimized_permute_reshape