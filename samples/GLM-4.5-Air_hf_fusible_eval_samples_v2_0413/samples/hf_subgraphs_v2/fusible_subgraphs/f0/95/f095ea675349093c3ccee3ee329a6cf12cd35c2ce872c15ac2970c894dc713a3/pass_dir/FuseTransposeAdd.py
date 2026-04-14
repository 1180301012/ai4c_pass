import torch
import triton
import triton.language as tl

def pattern(in_3, tmp_5):
    """Pattern: transpose + addition fusion"""
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    return tmp_7

def replacement_args(in_3, tmp_5):
    return (in_3, tmp_5)

@triton.jit
def fused_transpose_add_kernel(
    in_3_ptr,
    tmp_5_ptr,
    out_ptr,
    N_batch,
    N_channels,
    L,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for transpose + addition operations"""
    # Calculate program indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate offsets
    # in_3: [N_batch, N_channels, L] - we need [batch_idx, seq_idx, channel_idx]
    in_3_offset = batch_idx * N_channels * L + seq_idx * N_channels + channel_idx
    
    # tmp_5: [N_batch, N_channels, L-1] after conv1d+slice, but transposed to [N_batch, L-1, N_channels]
    # we need [batch_idx, seq_idx, channel_idx] where seq_idx < L-1
    tmp_5_offset = batch_idx * (L-1) * N_channels + seq_idx * N_channels + channel_idx
    
    # Load values
    in_3_val = tl.load(in_3_ptr + in_3_offset, mask=(batch_idx < N_batch) & (seq_idx < L) & (channel_idx < N_channels), other=0.0)
    tmp_5_val = tl.load(tmp_5_ptr + tmp_5_offset, mask=(batch_idx < N_batch) & (seq_idx < L-1) & (channel_idx < N_channels), other=0.0)
    
    # Perform addition where both tensors are valid
    seq_idx_mask = seq_idx < (L-1)  # only add for valid positions after transpose+slicing
    out_val = tl.where(seq_idx_mask, in_3_val + tmp_5_val, in_3_val)
    
    # Store result
    out_offset = batch_idx * N_channels * L + seq_idx * N_channels + channel_idx
    tl.store(out_ptr + out_offset, out_val, mask=(batch_idx < N_batch) & (seq_idx < L) & (channel_idx < N_channels))

@torch.fx.wrap
def fused_transpose_add(in_3, tmp_5):
    """Wrapper for fused transpose + addition operation"""
    N_batch, N_channels, L = in_3.shape
    
    # tmp_5 after conv1d+slice has shape [N_batch, N_channels, L-1]
    # After transpose, it becomes [N_batch, L-1, N_channels]
    # We need to match dimensions for addition with in_3 [N_batch, N_channels, L]
    
    # Create output tensor with same shape as in_3
    out = torch.empty_like(in_3)
    
    BLOCK_SIZE = 256
    
    # Launch kernel - need to handle the different dimensions carefully
    # We'll iterate through all positions and handle the transpose implicitly
    grid = (N_batch, N_channels, L)
    
    fused_transpose_add_kernel[grid](
        in_3, tmp_5, out,
        N_batch, N_channels, L,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_transpose_add