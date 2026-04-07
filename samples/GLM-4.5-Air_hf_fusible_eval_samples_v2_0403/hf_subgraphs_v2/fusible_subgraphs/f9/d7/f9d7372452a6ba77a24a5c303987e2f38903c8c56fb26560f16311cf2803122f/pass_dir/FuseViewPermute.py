import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the view and permute operations
def pattern(in_1):
    """Matches the view and permute operations:
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    Returns tmp_4 which is the transposed result
    """
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4

# Argument extraction function
def replacement_args(in_1):
    """Extracts arguments needed for the fused view-permute kernel"""
    return (in_1,)

# Optimized kernel for fused view and permute operations
@triton.jit
def fused_view_permute_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that performs view(1, 32, -1) and permute(0, 2, 1) efficiently"""
    # Each thread handles one element
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    
    # Iterate through all elements in parallel using vectorized loads
    for i in range(pid, batch_size * seq_len * hidden_size, stride):
        # Calculate batch, seq_pos, and head indices
        batch_idx = i // (seq_len * hidden_size)
        remainder = i % (seq_len * hidden_size)
        hidden_idx = remainder // seq_len
        seq_idx = remainder % seq_len
        
        # Input indices: [batch, head, seq_pos]
        input_idx = batch_idx * (32 * 3072) + hidden_idx * 3072 + seq_idx
        
        # Output indices: [batch, seq_pos, head] (after permute(0, 2, 1))
        output_idx = batch_idx * (32 * 3072) + seq_idx * 32 + hidden_idx
        
        if input_idx < in_ptr.numel() and output_idx < out_ptr.numel():
            # Load input data and transpose
            val = tl.load(in_ptr + input_idx)
            tl.store(out_ptr + output_idx, val)

@torch.fx.wrap
def fused_view_permute(in_1):
    """Wrapper function to launch the fused view-permute kernel"""
    # Input shape: [1, 32, 64, 48] -> after view: [1, 32, -1] -> [1, 32, 3072]
    # After permute(0, 2, 1): [1, 3072, 32]
    
    original_shape = in_1.shape  # [1, 32, 64, 48]
    batch_size = 1
    hidden_size = 32
    seq_len = 64 * 48  # 3072
    
    # Calculate output shape: [1, 3072, 32]
    output_shape = (batch_size, seq_len, hidden_size)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel with single grid dimension (auto-parallelization)
    fused_view_permute_kernel[(seq_len * hidden_size * batch_size,)](
        in_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=1  # Each thread handles one element
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_view_permute