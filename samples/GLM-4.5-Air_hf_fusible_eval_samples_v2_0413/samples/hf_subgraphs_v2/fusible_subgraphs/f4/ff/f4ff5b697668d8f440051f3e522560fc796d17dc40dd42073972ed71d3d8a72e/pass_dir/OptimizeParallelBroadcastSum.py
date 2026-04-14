import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation pattern
def pattern(tmp_2, in_0, in_1):
    # Pattern matches the entire computation sequence
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(tmp_2.shape[0], 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(tmp_2.shape[0], 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_3, tmp_10

# Argument extraction function
def replacement_args(tmp_2, in_0, in_1):
    return (tmp_2, in_0, in_1)

@triton.jit
def optimized_kernel(
    tmp_2_ptr,           # Softmax output [N, 17, 4096]
    in_0_ptr,           # Broadcast tensor [1, 1, 1, 64] 
    in_1_ptr,           # Broadcast tensor [1, 1, 64, 1]
    out_3_ptr,          # Reshaped spatial output [N, 17, 64, 64]
    out_10_ptr,         # Concatenated sum output [N, 17, 2]  
    N,                  # Batch size (tmp_2.shape[0])
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs for 3D grid: [batch, height, width]
    pid_batch = tl.program_id(0)
    pid_height = tl.program_id(1) 
    pid_width = tl.program_id(2)
    
    # Spatial offsets within block
    height_offsets = pid_height * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    width_offsets = pid_width * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Batch window (process up to 64 batches per program)
    batch_start = pid_batch * 64
    batch_indices = batch_start + tl.arange(0, 64)
    
    # Masks for bounds checking
    height_mask = height_offsets < 64
    width_mask = width_offsets < 64
    batch_mask = batch_indices < N
    
    # Load broadcast values (these are static for the program)
    in_0_val = tl.load(in_0_ptr, mask=True)  # Scalar broadcast for in_0
    in_1_vals = tl.load(in_1_ptr + tl.arange(0, 64, dtype=tl.int64), mask=True)  # 64 values for in_1 height-wise broadcast
    
    # Calculate number of valid batches in this window
    num_batches = tl.sum(batch_mask.to(tl.int32))
    
    # Initialize accumulators for sums  
    sum_x_accum = tl.zeros((num_batches, 17), dtype=tl.float32)
    sum_y_accum = tl.zeros((num_batches, 17), dtype=tl.float32)
    
    # For each feature dimension
    for feat_idx in range(17):
        # For each spatial position in this block
        for h_idx in height_offsets:
            for w_idx in width_offsets:
                if not (h_idx < 64 and w_idx < 64):
                    continue
                
                spatial_pos = h_idx * 64 + w_idx
                
                # Process each batch in window
                batch_idx = 0
                for curr_batch in batch_indices:
                    if curr_batch >= N:
                        continue
                    
                    # Load softmax value
                    src_offset = (curr_batch * 17 * 4096 + 
                                 feat_idx * 4096 + 
                                 spatial_pos)
                    
                    data_val = tl.load(tmp_2_ptr + src_offset, other=0.0).to(tl.float32)
                    
                    # Accumulate for in_0 path (spatial broadcast)
                    sum_x_accum[batch_idx, feat_idx] += data_val * in_0_val
                    
                    # Accumulate for in_1 path (height-wise broadcast)
                    sum_y_accum[batch_idx, feat_idx] += data_val * in_1_vals[h_idx]
                    
                    batch_idx += 1
        
        # Store spatial data (reshape: 4096 -> 64x64)
        batch_idx = 0
        for curr_batch in batch_indices:
            if curr_batch >= N:
                continue
            
            # Source: flattened softmax output
            src_base = curr_batch * 17 * 4096 + feat_idx * 4096
            
            # Target: spatial layout [N, 17, 64, 64] 
            tgt_base = curr_batch * 17 * 64 * 64 + feat_idx * 64 * 64
            
            # Copy spatial data
            for h_idx in height_offsets:
                for w_idx in width_offsets:
                    src_pos = src_base + h_idx * 64 + w_idx
                    tgt_pos = tgt_base + h_idx * 64 + w_idx
                    
                    if h_idx < 64 and w_idx < 64:
                        tl.store(out_3_ptr + tgt_pos, tl.load(tmp_2_ptr + src_pos, other=0.0))
            batch_idx += 1
    
    # Store final concatenated sums [N, 17, 2]
    batch_idx = 0
    for curr_batch in batch_indices:
        if curr_batch >= N:
            continue
            
        for feat_idx in range(17):
            base_offset = curr_batch * 17 * 2 + feat_idx * 2
            
            # Store in_0 sum
            tl.store(out_10_ptr + base_offset, sum_x_accum[batch_idx, feat_idx])
            # Store in_1 sum  
            tl.store(out_10_ptr + base_offset + 1, sum_y_accum[batch_idx, feat_idx])
        
        batch_idx += 1

# Kernel wrapper
@torch.fx.wrap  
def compute_optimized(tmp_2, in_0, in_1):
    N = tmp_2.shape[0]
    
    # Output tensors
    out_3 = torch.empty(N, 17, 64, 64, dtype=tmp_2.dtype, device=tmp_2.device)
    out_10 = torch.empty(N, 17, 2, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Launch Triton kernel
    optimized_kernel[(17, 64, (N + 63) // 64)](
        tmp_2_ptr=tmp_2,
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_3_ptr=out_3,
        out_10_ptr=out_10,
        N=N,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
    )
    
    return out_3, out_10

# Replacement function
def replacement_func():
    return compute_optimized