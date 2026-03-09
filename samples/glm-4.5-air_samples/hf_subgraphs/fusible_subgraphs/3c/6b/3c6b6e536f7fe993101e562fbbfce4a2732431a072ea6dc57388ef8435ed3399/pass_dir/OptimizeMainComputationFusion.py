import torch
import triton
import triton.language as tl

# Pattern matching function for the main computation block
def pattern(in_6, in_5, in_2, in_4):
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4

# Argument extraction function
def replacement_args(in_6, in_5, in_2, in_4):
    return (in_6, in_5, in_2, in_4)

# Optimized kernel for fused computation
@triton.jit
def fused_compute_kernel(
    in_6_ptr, in_5_ptr, in_2_ptr, in_4_ptr, out_ptr,
    batch_size, seq_len, hidden_dim,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Calculate program IDs
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Process all hidden dimensions in one go
    hidden_offsets = tl.arange(0, BLOCK_SIZE_Y)
    mask = hidden_offsets < hidden_dim
    
    # Calculate base pointers for this batch/seq position
    in_6_base = batch_id * seq_len * 64 + seq_id * 64
    in_5_base = batch_id * seq_len * 64 + seq_id * 64
    in_2_base = seq_id * 64
    in_4_base = batch_id * seq_len * 64 + seq_id * 64
    
    # Load input tensors with proper masking
    x2_part = tl.load(in_2_ptr + in_2_base + hidden_offsets, mask=mask, other=0.0)
    in_5_part = tl.load(in_5_ptr + in_5_base + hidden_offsets, mask=mask, other=0.0)
    in_6_part = tl.load(in_6_ptr + in_6_base + hidden_offsets, mask=mask, other=0.0)
    in_4_part = tl.load(in_4_ptr + in_4_base + hidden_offsets, mask=mask, other=0.0)
    
    # Perform fused computation:
    # tmp_0 = -in_6
    # tmp_1 = cat(tmp_0, in_5) -> concatenates [-in_6_part, in_5_part] 
    # tmp_2 = tmp_1 * in_2 -> in_2_part has the right shape for broadcasting
    # tmp_3 = in_4 + tmp_2
    # tmp_4 = tmp_3.to(float32) -> already float32
    
    # Apply negation to in_6 first half (32 elements) [0, 31]
    # For positions 0-31, we use -in_6_part, for positions 32-63, we use in_5_part
    result_first_half = -in_6_part
    result_second_half = in_5_part
    
    # Combine using masks for each half
    first_half_mask = hidden_offsets < 32
    second_half_mask = hidden_offsets >= 32
    final_result = tl.where(first_half_mask, result_first_half, result_second_half)
    
    # Multiply by in_2 (broadcasting)
    tmp_2 = final_result * x2_part
    
    # Add in_4
    out_val = in_4_part + tmp_2
    
    # Store results
    tl.store(out_ptr + in_4_base + hidden_offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_compute(in_6, in_5, in_2, in_4):
    # Get tensor shapes
    batch_size, seq_len, _, _ = in_4.shape
    
    # Create output tensor
    out = torch.empty_like(in_4)
    
    # Set up Triton kernel launch parameters
    BLOCK_SIZE_Y = 64  # Process full hidden dimension per program
    hidden_dim = in_4.shape[-1]  # Could be 64 or different for other graphs
    
    # Create grid: (batch_size, seq_len) - each program handles one batch and sequence position
    grid = (batch_size, seq_len)
    
    # Launch kernel
    fused_compute_kernel[grid](
        in_6_ptr=in_6,
        in_5_ptr=in_5,
        in_2_ptr=in_2,
        in_4_ptr=in_4,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE_X=1,  # Unused but required
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_compute