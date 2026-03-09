import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern matches: (in_0 + in_1) with slice from in_2 OR
    #                  (in_0 + in_2) with slice from in_1
    tmp_0 = in_0 + in_1
    tmp_1 = in_2[slice(None, None, None), slice(in_0.shape[1], None, None)]
    result = torch.cat([tmp_0, tmp_1], dim=1)
    return result

def replacement_args(in_0, in_1, in_2):
    # Extract slice start index from the first tensor's channel dimension
    slice_start = in_0.shape[1]
    return (in_0, in_1, in_2, slice_start)

@triton.jit
def fusion_kernel(
    # Input pointers
    in0_ptr, in1_ptr, in2_ptr,
    # Output pointer  
    out_ptr,
    # Tensor shapes
    batch_size, c0, h, w, c2, slice_idx,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Calculate program ids for each dimension
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Calculate offsets within each block
    batch_offsets = batch_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    channel_offsets = channel_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create bounds masks
    batch_mask = batch_offsets < batch_size
    channel_mask = channel_offsets < c2
    
    # Loop through each batch and channel in this block
    for b in range(BLOCK_SIZE_M):
        if batch_offsets[b] >= batch_size:
            continue
            
        for c in range(BLOCK_SIZE_N):
            if channel_offsets[c] >= c2:
                continue
                
            # Compute flat offset for this position
            out_offset = (batch_offsets[b] * c2 * h * w + 
                         channel_offsets[c] * h * w + 
                         tl.arange(0, h) * w + 
                         tl.arange(0, w))
            
            h_mask = tl.arange(0, h) < h
            w_mask = tl.arange(0, w) < w
            mask = h_mask[:, None] & w_mask[None, :]
            
            # Determine which operation to perform based on channel position
            if channel_offsets[c] < slice_idx:
                # This case shouldn't happen if our analysis is correct
                continue
            elif channel_offsets[c] < slice_idx + c0:
                # First part: in0 + in1 (for channels 0 to c0-1)
                # Map to position in in0/in1: pos = channel_offset - slice_idx
                pos_in_01 = channel_offsets[c] - slice_idx
                in0_offset = (batch_offsets[b] * c0 * h * w + 
                             pos_in_01 * h * w + 
                             tl.arange(0, h) * w + 
                             tl.arange(0, w))
                in1_offset = in0_offset  # Same offset since in0 and in1 have same shape
                
                in0_val = tl.load(in0_ptr + in0_offset, mask=mask, other=0.0)
                in1_val = tl.load(in1_ptr + in1_offset, mask=mask, other=0.0)
                
                # Add and store
                result = in0_val + in1_val
                tl.store(out_ptr + out_offset, result, mask=mask)
            else:
                # Second part: in2 slice (for channels slice_idx to end)
                # Here we use the channel offset directly for in2
                in2_channel = channel_offsets[c]
                in2_offset = (batch_offsets[b] * c2 * h * w + 
                             in2_channel * h * w + 
                             tl.arange(0, h) * w + 
                             tl.arange(0, w))
                
                in2_val = tl.load(in2_ptr + in2_offset, mask=mask, other=0.0)
                tl.store(out_ptr + out_offset, in2_val, mask=mask)

@torch.fx.wrap
def fused_add_slice_concat(x1, x2, x3, slice_start):
    # Get tensor shapes
    batch_size, c0, h, w = x1.shape
    _, c2, _, _ = x3.shape
    
    # Create output tensor
    out = torch.empty((batch_size, c2, h, w), dtype=torch.float32, device=x1.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_M = 8   # Process 8 batches at a time (to limit total threads)
    BLOCK_SIZE_N = 32  # Process 32 channels at a time
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (c2 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fusion_kernel[grid_m, grid_n](
        in0_ptr=x1,
        in1_ptr=x2, 
        in2_ptr=x3,
        out_ptr=out,
        batch_size=batch_size,
        c0=c0, h=h, w=w, c2=c2, slice_idx=slice_start,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_add_slice_concat