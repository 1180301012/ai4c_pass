import torch
import triton
import triton.language as tl


# Pattern matching function - match transpose + add
def pattern(tmp_6, tmp_3):
    """
    Match the pattern: transpose(1,2) + add
    
    tmp_7 = tmp_6.transpose(1, 2)  # (B, C, N) -> (B, N, C)
    tmp_8 = tmp_7 + tmp_3          # (B, N, C) + (B, N, C) -> (B, N, C)
    """
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = tmp_7 + tmp_3
    return tmp_8

# Argument extraction function  
def replacement_args(tmp_6, tmp_3):
    return (tmp_6, tmp_3)


# Fixed block size - no autotuning overhead
@triton.jit
def fused_transpose_add_kernel(
    input_ptr,      # tmp_6: (B, C, N) - output of flatten
    pos_embed_ptr,  # tmp_3: (B, N, C) - positional embedding
    output_ptr,     # Output: (B, N, C)
    B: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
):
    # Each program processes all channels for one (batch, position) pair
    # Grid: (B * N,)
    pid = tl.program_id(0)
    batch_idx = pid // N
    pos_idx = pid % N
    
    # Calculate base offset for this (batch, position)
    base_offset = batch_idx * N * C + pos_idx * C
    
    # Fixed BLOCK_SIZE = 256 (good for C=768)
    BLOCK_SIZE: tl.constexpr = 256
    
    # Process channels in blocks
    for ch in range(0, C, BLOCK_SIZE):
        # Create offsets for this block
        ch_offsets = ch + tl.arange(0, BLOCK_SIZE)
        mask = ch_offsets < C
        
        # Load from input (B, C, N) -> input[batch, ch, pos]
        input_offsets = batch_idx * C * N + ch_offsets * N + pos_idx
        input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Load from pos_embed (B, N, C) -> pos_embed[batch, pos, ch]
        pos_offsets = base_offset + ch_offsets
        pos_vals = tl.load(pos_embed_ptr + pos_offsets, mask=mask, other=0.0)
        
        # Add and store
        out_vals = input_vals + pos_vals
        tl.store(output_ptr + pos_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_transpose_add_kernel_wrapper(tmp_6, tmp_3):
    """
    Fused implementation: transpose + add in a single kernel.
    Fixed block size for minimal overhead.
    """
    B = tmp_6.size(0)
    C = tmp_6.size(1)
    N = tmp_6.size(2)
    
    # Create output tensor
    output = torch.empty(B, N, C, dtype=tmp_6.dtype, device=tmp_6.device)
    
    # Grid: one program per (batch, position) pair
    grid = (B * N,)
    
    fused_transpose_add_kernel[grid](
        tmp_6,
        tmp_3,
        output,
        B, C, N,
    )
    
    return output


def replacement_func():
    return fused_transpose_add_kernel_wrapper