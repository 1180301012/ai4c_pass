import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern matching the target computation sequence"""
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

def replacement_args(in_0):
    """Extract arguments needed for replacement"""
    return (in_0,)

@triton.jit
def optimized_cumsum_mask_kernel(
    input_ptr,
    output_ptr,
    n_seqs,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel that fuses cumsum + mask operations"""
    # Program ID determines which sequence this program handles
    seq_id = tl.program_id(0)
    # Start offset for this sequence
    seq_offset = seq_id * seq_len
    
    # Create offset range for this sequence
    offsets = seq_offset + tl.arange(0, BLOCK_SIZE)
    # Mask to handle boundary conditions
    mask = offsets < seq_offset + seq_len
    
    # Load input elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Create mask: 1 where input != 1, 0 where input == 1
    mask_vals = (input_vals != 1).to(tl.int32)
    
    # Compute cumulative sum within the block
    cumsum_values = tl.cumsum(mask_vals, 0)
    
    # Apply final computation: cumsum * mask + 1
    # This implements the exact logic from the original computation:
    # tmp_1 = in_0.ne(1), tmp_2 = tmp_1.int(), tmp_3 = cumsum(tmp_2)
    # tmp_6 = tmp_5 * tmp_2 (where tmp_5 is cumsum after redundant ops +0)
    # tmp_8 = tmp_7 + 1 = tmp_6 + 1
    result_values = cumsum_values * mask_vals + 1
    
    # Store result
    tl.store(output_ptr + offsets, result_values, mask=mask)

@torch.fx.wrap
def optimized_cumsum_mask_fusion(in_0):
    """Wrapper function to launch the optimized kernel"""
    # Get input dimensions
    n_seqs, seq_len = in_0.shape
    
    # Create output tensor
    out = torch.empty_like(in_0, dtype=torch.long)
    
    # Use optimal fixed block size for balanced performance
    BLOCK_SIZE = 1024  # Best balance across all sequence lengths
    
    # Calculate grid dimensions
    num_seqs = n_seqs
    num_blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel using 2D grid: (num_seqs, num_blocks_per_seq)
    optimized_cumsum_mask_kernel[(num_seqs, num_blocks_per_seq)](
        input_ptr=in_0,
        output_ptr=out,
        n_seqs=n_seqs,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_cumsum_mask_fusion